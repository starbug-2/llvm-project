//===- StarbugVLIWMachineScheduler.cpp - Starbug pre-RA VLIW scheduler ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StarbugVLIWMachineScheduler.h"
#include "StarbugVLIWIssueRules.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::RISCVVLIW;

#define DEBUG_TYPE "machine-scheduler"

// Below PriorityTwo (50). Fitting the current packet and excess register
// pressure still decide first. These only reorder candidates that are
// otherwise close.
static constexpr int PointerBumpBonus = 15;
static constexpr int MacMulAfterLoad = 30;
static constexpr int MacAddAfterMul = 34;
static constexpr int MacStoreAfterAdd = 18;

// Pseudos that do not occupy an issue slot. COPY does: it is a move.
static bool consumesIssueSlot(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case TargetOpcode::EXTRACT_SUBREG:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::REG_SEQUENCE:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::CFI_INSTRUCTION:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::INLINEASM:
  case TargetOpcode::INLINEASM_BR:
    return false;
  default:
    return !MI.isDebugInstr();
  }
}

static int assignLane(const StarbugVLIWConfig &Config, OpClass Class,
                      ArrayRef<unsigned> UsedLanes) {
  const unsigned NumLanes = Config.maxBundleWidth();
  auto Busy = [&](unsigned Lane) {
    for (unsigned Used : UsedLanes)
      if (Used == Lane)
        return true;
    return false;
  };

  // Prefer non-zero lanes so lane 0 stays free for an unrestricted opcode.
  if (Config.Scheduler.ReserveLane0ForAny && NumLanes > 1) {
    for (unsigned Lane = 1; Lane < NumLanes; ++Lane) {
      if (Config.isLaneLegal(Lane, Class) && !Busy(Lane))
        return static_cast<int>(Lane);
    }
    if (Config.isLaneLegal(0, Class) && !Busy(0))
      return 0;
    return -1;
  }

  for (unsigned Lane = 0; Lane < NumLanes; ++Lane) {
    if (Config.isLaneLegal(Lane, Class) && !Busy(Lane))
      return static_cast<int>(Lane);
  }
  return -1;
}

static bool scheduledDataPred(const SUnit *SU, OpClass Want, bool PacketizeFP) {
  for (const SDep &Dep : SU->Preds) {
    if (Dep.getKind() != SDep::Data)
      continue;
    const SUnit *Pred = Dep.getSUnit();
    if (!Pred->isScheduled || !Pred->isInstr())
      continue;
    if (classifyOpClass(*Pred->getInstr(), PacketizeFP) == Want)
      return true;
  }
  return false;
}

// Bottom-up, the consumer is the node already placed. The bonus is the same
// pairing, seen from the other end of the edge.
static bool scheduledDataSucc(const SUnit *SU, OpClass Want, bool PacketizeFP) {
  for (const SDep &Dep : SU->Succs) {
    if (Dep.getKind() != SDep::Data)
      continue;
    const SUnit *Succ = Dep.getSUnit();
    if (!Succ->isScheduled || !Succ->isInstr())
      continue;
    if (classifyOpClass(*Succ->getInstr(), PacketizeFP) == Want)
      return true;
  }
  return false;
}

StarbugVLIWResourceModel::StarbugVLIWResourceModel(
    const TargetSubtargetInfo &STI, const TargetSchedModel *SchedModel)
    : VLIWResourceModel(STI, SchedModel), Config(StarbugVLIWConfig::fromCommandLine()) {
}

std::optional<MemBank>
StarbugVLIWResourceModel::cachedBank(const MachineInstr &MI) {
  BankCacheEntry &Entry = BankCache[&MI];
  if (!Entry.Computed) {
    Entry.Computed = true;
    if (std::optional<MemBank> Bank = memoryBank(MI, Config.Bank1Skew)) {
      Entry.Known = true;
      Entry.Bank = *Bank;
    }
  }
  if (!Entry.Known)
    return std::nullopt;
  return Entry.Bank;
}

bool StarbugVLIWResourceModel::lanesAllow(const MachineInstr &Cand) const {
  const bool PacketizeFP = Config.Scheduler.PacketizeFP;
  SmallVector<unsigned, 8> Used;
  for (const SUnit *SU : Packet) {
    const MachineInstr *MI = SU->getInstr();
    if (!MI || !consumesIssueSlot(*MI))
      continue;
    const int Lane =
        assignLane(Config, classifyOpClass(*MI, PacketizeFP), Used);
    if (Lane >= 0)
      Used.push_back(static_cast<unsigned>(Lane));
  }
  return assignLane(Config, classifyOpClass(Cand, PacketizeFP), Used) >= 0;
}

bool StarbugVLIWResourceModel::banksAllow(const MachineInstr &Cand) {
  SmallVector<MachineInstr *, 8> Mems;
  for (SUnit *SU : Packet) {
    MachineInstr *MI = SU->getInstr();
    if (MI && (MI->mayLoad() || MI->mayStore()))
      Mems.push_back(MI);
  }
  return memoryBanksAllow(Mems, Cand, [&](const MachineInstr &MI) {
    return cachedBank(MI);
  });
}

bool StarbugVLIWResourceModel::isResourceAvailable(SUnit *SU, bool IsTop) {
  if (!SU || !SU->getInstr())
    return false;
  const MachineInstr &MI = *SU->getInstr();
  if (consumesIssueSlot(MI)) {
    if (!lanesAllow(MI))
      return false;
    if ((MI.mayLoad() || MI.mayStore()) && !banksAllow(MI))
      return false;
  }
  return VLIWResourceModel::isResourceAvailable(SU, IsTop);
}

StarbugConvergingVLIWScheduler::StarbugConvergingVLIWScheduler()
    : Config(StarbugVLIWConfig::fromCommandLine()) {}

VLIWResourceModel *StarbugConvergingVLIWScheduler::createVLIWResourceModel(
    const TargetSubtargetInfo &STI, const TargetSchedModel *SchedModel) const {
  return new StarbugVLIWResourceModel(STI, SchedModel);
}

int StarbugConvergingVLIWScheduler::SchedulingCost(ReadyQueue &Q, SUnit *SU,
                                                   SchedCandidate &Candidate,
                                                   RegPressureDelta &Delta,
                                                   bool verbose) {
  // PriorityTwo is the "fits in this packet" bonus. Stay under it.
  static_assert(PointerBumpBonus < PriorityTwo);
  static_assert(MacMulAfterLoad < PriorityTwo);
  static_assert(MacAddAfterMul < PriorityTwo);
  static_assert(MacStoreAfterAdd < PriorityTwo);

  int ResCount =
      ConvergingVLIWScheduler::SchedulingCost(Q, SU, Candidate, Delta, verbose);
  if (!SU || SU->isScheduled || !SU->isInstr())
    return ResCount;

  const MachineInstr &MI = *SU->getInstr();
  const bool PacketizeFP = Config.Scheduler.PacketizeFP;
  const bool TopDown = Q.getID() == TopQID;
  int Bonus = 0;

  if (Config.Scheduler.PrioritizePointerBumps && isPointerBump(MI))
    Bonus += PointerBumpBonus;

  if (Config.Scheduler.PreferMACOverlap) {
    const OpClass Class = classifyOpClass(MI, PacketizeFP);
    if (TopDown) {
      if (Class == OpClass::ALUMulDiv &&
          scheduledDataPred(SU, OpClass::Load, PacketizeFP))
        Bonus += MacMulAfterLoad;
      else if (Class == OpClass::ALUAddSub &&
               scheduledDataPred(SU, OpClass::ALUMulDiv, PacketizeFP))
        Bonus += MacAddAfterMul;
      else if (Class == OpClass::Store &&
               scheduledDataPred(SU, OpClass::ALUAddSub, PacketizeFP))
        Bonus += MacStoreAfterAdd;
    } else {
      if (Class == OpClass::Load &&
          scheduledDataSucc(SU, OpClass::ALUMulDiv, PacketizeFP))
        Bonus += MacMulAfterLoad;
      else if (Class == OpClass::ALUMulDiv &&
               scheduledDataSucc(SU, OpClass::ALUAddSub, PacketizeFP))
        Bonus += MacAddAfterMul;
      else if (Class == OpClass::ALUAddSub &&
               scheduledDataSucc(SU, OpClass::Store, PacketizeFP))
        Bonus += MacStoreAfterAdd;
    }
  }

  if (Bonus && verbose)
    LLVM_DEBUG(dbgs() << " bump/mac+" << Bonus << '|');
  return ResCount + Bonus;
}

ScheduleDAGInstrs *llvm::createStarbugVLIWMachineSched(MachineSchedContext *C) {
  return new VLIWMachineScheduler(
      C, std::make_unique<StarbugConvergingVLIWScheduler>());
}

static MachineSchedRegistry
    StarbugMISchedRegistry("starbug", "Run Starbug's VLIW pre-RA scheduler",
                           createStarbugVLIWMachineSched);
