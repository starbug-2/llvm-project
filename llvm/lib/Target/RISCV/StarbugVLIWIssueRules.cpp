//===- StarbugVLIWIssueRules.cpp - Lane and bank rules for Starbug --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StarbugVLIWIssueRules.h"
#include "RISCVInstrInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;
using namespace llvm::RISCVVLIW;

static bool isFPRegClass(const TargetRegisterClass *RC) {
  return RISCV::FPR16RegClass.hasSubClassEq(RC) ||
         RISCV::FPR32RegClass.hasSubClassEq(RC) ||
         RISCV::FPR64RegClass.hasSubClassEq(RC) ||
         RISCV::FPR128RegClass.hasSubClassEq(RC);
}

// Physical registers answer this after RA. Virtual registers answer it before
// RA, which is when the VLIW scheduler runs. An opcode list would miss a new
// FP instruction; the register file does not.
static bool touchesFPRegisters(const MachineInstr &MI) {
  const MachineRegisterInfo *MRI =
      MI.getParent() ? &MI.getMF()->getRegInfo() : nullptr;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();
    if (!Reg)
      continue;
    if (Reg.isPhysical()) {
      if (RISCV::FPR16RegClass.contains(Reg) ||
          RISCV::FPR32RegClass.contains(Reg) ||
          RISCV::FPR64RegClass.contains(Reg) ||
          RISCV::FPR128RegClass.contains(Reg))
        return true;
      continue;
    }
    if (MRI && Reg.isVirtual() && isFPRegClass(MRI->getRegClass(Reg)))
      return true;
  }
  return false;
}

// fdiv and fsqrt drive FDivBusyE, which the core ORs into an execute stall.
// They stay legal in a worker lane; the class exists so a sweep can exclude
// them.
static bool isFPDivSqrtOpcode(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::FDIV_S:
  case RISCV::FDIV_D:
  case RISCV::FDIV_H:
  case RISCV::FSQRT_S:
  case RISCV::FSQRT_D:
  case RISCV::FSQRT_H:
    return true;
  default:
    return false;
  }
}

OpClass RISCVVLIW::classifyOpClass(const MachineInstr &MI, bool PacketizeFP) {
  // These predicates come from the instruction description and outrank any
  // opcode guess.
  if (MI.isBranch() || MI.isTerminator() || MI.isCall() || MI.isReturn())
    return OpClass::Branch;
  if (MI.mayLoad())
    return OpClass::Load;
  if (MI.mayStore())
    return OpClass::Store;

  // Floating point belongs in the worker lanes. FP loads and stores are not
  // covered here: mayLoad/mayStore above already claimed them.
  if (touchesFPRegisters(MI))
    return !PacketizeFP ? OpClass::Any
                        : (isFPDivSqrtOpcode(MI.getOpcode()) ? OpClass::FPUDivSqrt
                                                             : OpClass::FPU);

  switch (MI.getOpcode()) {
  case RISCV::ADD:
  case RISCV::ADDI:
  case RISCV::SUB:
  case RISCV::ADDW:
  case RISCV::ADDIW:
  case RISCV::SUBW:
  case RISCV::C_ADD:
  case RISCV::C_ADDI:
  case RISCV::C_ADDI16SP:
  case RISCV::C_ADDI4SPN:
  case RISCV::C_SUB:
  case RISCV::C_ADDW:
  case RISCV::C_SUBW:
    return OpClass::ALUAddSub;

  case RISCV::MUL:
  case RISCV::MULH:
  case RISCV::MULHU:
  case RISCV::MULHSU:
  case RISCV::MULW:
  case RISCV::DIV:
  case RISCV::DIVU:
  case RISCV::DIVW:
  case RISCV::DIVUW:
  case RISCV::REM:
  case RISCV::REMU:
  case RISCV::REMW:
  case RISCV::REMUW:
    return OpClass::ALUMulDiv;

  case RISCV::SLL:
  case RISCV::SLLI:
  case RISCV::SRL:
  case RISCV::SRLI:
  case RISCV::SRA:
  case RISCV::SRAI:
  case RISCV::SLLW:
  case RISCV::SLLIW:
  case RISCV::SRLW:
  case RISCV::SRLIW:
  case RISCV::SRAW:
  case RISCV::SRAIW:
  case RISCV::C_SLLI:
  case RISCV::C_SRLI:
  case RISCV::C_SRAI:
    return OpClass::ALUShift;

  case RISCV::XOR:
  case RISCV::XORI:
  case RISCV::OR:
  case RISCV::ORI:
  case RISCV::AND:
  case RISCV::ANDI:
  case RISCV::C_XOR:
  case RISCV::C_OR:
  case RISCV::C_AND:
  case RISCV::C_ANDI:
    return OpClass::ALULogic;

  case RISCV::SLT:
  case RISCV::SLTI:
  case RISCV::SLTU:
  case RISCV::SLTIU:
    return OpClass::Compare;

  case RISCV::LUI:
  case RISCV::C_LUI:
  case RISCV::C_LI:
  case RISCV::C_MV:
  case TargetOpcode::COPY:
    return OpClass::Move;

  case RISCV::CSRRW:
  case RISCV::CSRRS:
  case RISCV::CSRRC:
  case RISCV::CSRRWI:
  case RISCV::CSRRSI:
  case RISCV::CSRRCI:
    return OpClass::CSR;

  default:
    return OpClass::Any;
  }
}

bool RISCVVLIW::isPointerBump(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::ADDI:
  case RISCV::ADDIW:
  case RISCV::C_ADDI:
  case RISCV::C_ADDIW:
  case RISCV::C_ADDI16SP:
  case RISCV::C_ADDI4SPN:
    break;
  default:
    return false;
  }

  Register DefReg;
  Register UseReg;
  bool HasImm = false;
  int64_t Imm = 0;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && !DefReg)
      DefReg = MO.getReg();
    if (MO.isReg() && MO.isUse() && !UseReg)
      UseReg = MO.getReg();
    if (MO.isImm()) {
      HasImm = true;
      Imm = MO.getImm();
    }
  }
  return DefReg && UseReg && DefReg == UseReg && HasImm && Imm != 0;
}

bool RISCVVLIW::banksDifferEveryIteration(MemBank A, MemBank B) {
  return A.Flip == B.Flip && A.Bit != B.Bit;
}

static std::optional<unsigned> bankFromSection(const GlobalValue &GV,
                                               int64_t Disp,
                                               unsigned Bank1Skew) {
  StringRef Section = GV.getSection();
  unsigned Bit = 0;
  if (Section == ".bank0" || Section.starts_with(".bank0."))
    Bit = 0;
  else if (Section == ".bank1" || Section.starts_with(".bank1."))
    Bit = (Bank1Skew >> 2) & 1u;
  else
    return std::nullopt;
  return Bit ^ static_cast<unsigned>((Disp >> 2) & 1);
}

static Register getMemoryBaseReg(const MachineInstr &MI) {
  for (int I = static_cast<int>(MI.getNumOperands()) - 1; I >= 0; --I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse())
      continue;
    Register Reg = MO.getReg();
    if (Reg)
      return Reg;
  }
  return Register();
}

// The last definition of Reg in MBB before Before. A null Before searches from
// the end of the block, including the last instruction.
static const MachineInstr *findDefBefore(const MachineBasicBlock &MBB,
                                         const MachineInstr *Before,
                                         Register Reg) {
  auto It = Before ? Before->getIterator() : MBB.end();
  while (It != MBB.begin()) {
    --It;
    for (const MachineOperand &MO : It->operands())
      if (MO.isReg() && MO.isDef() && MO.getReg() == Reg)
        return &*It;
  }
  return nullptr;
}

// A phi whose latch value is addi or c.addi of the phi by a nonzero multiple
// of 4. Incoming is the value from the single predecessor outside the block.
// StepImm is that latch immediate. The load uses the phi, so the step belongs
// to the next iteration and is not part of this iteration's displacement.
static bool phiPreheader(const MachineInstr &Phi, Register &Incoming,
                         MachineBasicBlock *&Pred, int64_t &StepImm) {
  if (!Phi.isPHI() || !Phi.getParent())
    return false;
  const MachineBasicBlock &Loop = *Phi.getParent();
  Incoming = Register();
  Pred = nullptr;
  Register Latch;
  bool SawLatch = false;
  for (unsigned I = 1; I + 1 < Phi.getNumOperands(); I += 2) {
    if (!Phi.getOperand(I).isReg() || !Phi.getOperand(I + 1).isMBB())
      return false;
    MachineBasicBlock *From = Phi.getOperand(I + 1).getMBB();
    if (From == &Loop) {
      if (SawLatch)
        return false;
      SawLatch = true;
      Latch = Phi.getOperand(I).getReg();
    } else {
      if (Pred)
        return false;
      Pred = From;
      Incoming = Phi.getOperand(I).getReg();
    }
  }
  if (!Pred || !Incoming || !SawLatch || !Latch)
    return false;
  const Register PhiReg = Phi.getOperand(0).getReg();
  const MachineInstr *Step = findDefBefore(Loop, nullptr, Latch);
  if (!Step || Step == &Phi)
    return false;
  const unsigned Opc = Step->getOpcode();
  Register Src;
  int64_t Imm = 0;
  if (Opc == RISCV::ADDI && Step->getNumOperands() >= 3 &&
      Step->getOperand(1).isReg() && Step->getOperand(2).isImm()) {
    Src = Step->getOperand(1).getReg();
    Imm = Step->getOperand(2).getImm();
  } else if (Opc == RISCV::C_ADDI && Step->getNumOperands() >= 2 &&
             Step->getOperand(0).isReg() && Step->getOperand(1).isImm()) {
    Src = Step->getOperand(0).getReg();
    Imm = Step->getOperand(1).getImm();
  } else {
    return false;
  }
  // A step that is not a multiple of 4 can carry into bit 2. Leave that
  // pointer unknown. A multiple of 4 toggles the bank each iteration exactly
  // when its own bit 2 is set.
  if (Imm == 0 || (Imm & 3) != 0 || Src != PhiReg)
    return false;
  StepImm = Imm;
  return true;
}

// addi / c.addi / a move. Copies are addi with a zero immediate.
static bool parseAddImm(const MachineInstr &MI, Register &Src, int64_t &Imm) {
  const unsigned Opc = MI.getOpcode();
  if (Opc == RISCV::ADDI && MI.getNumOperands() >= 3 &&
      MI.getOperand(1).isReg() && MI.getOperand(2).isImm()) {
    Src = MI.getOperand(1).getReg();
    Imm = MI.getOperand(2).getImm();
    return true;
  }
  if (Opc == RISCV::C_ADDI && MI.getNumOperands() >= 2 &&
      MI.getOperand(0).isReg() && MI.getOperand(1).isImm()) {
    Src = MI.getOperand(0).getReg();
    Imm = MI.getOperand(1).getImm();
    return true;
  }
  if ((Opc == RISCV::C_MV || MI.isCopy()) && MI.getNumOperands() >= 2 &&
      MI.getOperand(1).isReg()) {
    Src = MI.getOperand(1).getReg();
    Imm = 0;
    return true;
  }
  return false;
}

// Post-RA, a single-block loop carries a pointer with addi instead of a phi.
// The bump is the last definition of Reg, and its source is live in to the
// block. The sum is the bytes added once per iteration.
static std::optional<int64_t> selfLatchStep(const MachineBasicBlock &Block,
                                            Register Reg) {
  if (!Block.isSuccessor(&Block))
    return std::nullopt;
  Register Cur = Reg;
  int64_t Sum = 0;
  const MachineInstr *Cursor = nullptr;
  for (unsigned N = 0; N < 8; ++N) {
    const MachineInstr *Def = findDefBefore(Block, Cursor, Cur);
    if (!Def) {
      if (Cur == Reg && Sum != 0)
        return Sum;
      return std::nullopt;
    }
    Register Src;
    int64_t Imm = 0;
    if (!parseAddImm(*Def, Src, Imm))
      return std::nullopt;
    Sum += Imm;
    if (Src == Reg && !findDefBefore(Block, Def, Src))
      return Sum;
    Cur = Src;
    Cursor = Def;
  }
  return std::nullopt;
}

std::optional<MemBank> RISCVVLIW::memoryBank(const MachineInstr &Start,
                                             unsigned Bank1Skew) {
  const MachineFunction *MF =
      Start.getParent() ? Start.getParent()->getParent() : nullptr;
  int64_t Disp = 0;
  const GlobalValue *GV = nullptr;
  bool SawFI = false;
  for (const MachineOperand &MO : Start.operands()) {
    if (MO.isImm())
      Disp += MO.getImm();
    else if (MO.isGlobal()) {
      GV = MO.getGlobal();
      Disp += MO.getOffset();
    } else if (MO.isFI() && MF) {
      Disp += MF->getFrameInfo().getObjectOffset(MO.getIndex());
      SawFI = true;
    }
  }
  // Flip is filled in only for the loop that re-executes this instruction.
  // An early return below is a fixed address. A later self-loop whose step
  // toggles bit 2 is an outer index, and then the bank is not one value.
  unsigned Flip = 0;
  bool SawLoop = false;
  const MachineBasicBlock *LoadBlock = Start.getParent();
  auto withBit = [&](unsigned Bit) -> MemBank {
    return MemBank{Bit, Flip};
  };
  if (GV) {
    std::optional<unsigned> Bit = bankFromSection(*GV, Disp, Bank1Skew);
    if (!Bit)
      return std::nullopt;
    return withBit(*Bit);
  }
  if (SawFI)
    return withBit(static_cast<unsigned>((Disp >> 2) & 1));

  if (!LoadBlock)
    return std::nullopt;
  Register Base = getMemoryBaseReg(Start);
  const MachineInstr *Cursor = &Start;
  const MachineBasicBlock *Block = LoadBlock;
  for (unsigned Steps = 0; Steps < 24; ++Steps) {
    if (!Base || !Block)
      return std::nullopt;
    if (Base == RISCV::X2)
      return withBit(static_cast<unsigned>((Disp >> 2) & 1));
    const MachineInstr *Def = findDefBefore(*Block, Cursor, Base);
    if (!Def) {
      // No phi remains after register allocation. A single-block loop carries
      // the pointer in an addi on the back edge. That step is the next
      // iteration, so it is not part of this address. The other predecessor
      // is the value on iteration 0. A live-in that is not a loop bump is
      // the value at the end of the unique predecessor.
      const MachineBasicBlock *Entry = nullptr;
      std::optional<int64_t> Step = selfLatchStep(*Block, Base);
      if (Step) {
        if (*Step == 0 || (*Step & 3) != 0)
          return std::nullopt;
        if (*Step & 4) {
          if (SawLoop)
            return std::nullopt;
          Flip = 1;
        }
        SawLoop = true;
        for (const MachineBasicBlock *Pred : Block->predecessors()) {
          if (Pred == Block)
            continue;
          if (Entry)
            return std::nullopt;
          Entry = Pred;
        }
      } else {
        if (Block->pred_size() != 1)
          return std::nullopt;
        Entry = *Block->pred_begin();
      }
      if (!Entry)
        return std::nullopt;
      Cursor = nullptr;
      Block = Entry;
      continue;
    }

    if (Def->isPHI()) {
      Register Incoming;
      MachineBasicBlock *Pred = nullptr;
      int64_t StepImm = 0;
      if (!phiPreheader(*Def, Incoming, Pred, StepImm))
        return std::nullopt;
      const bool Toggles = (StepImm & 4) != 0;
      if (Def->getParent() == LoadBlock) {
        // This iteration's address is the preheader value plus the
        // displacement already accumulated. The step is the next iteration.
        Flip = Toggles ? 1u : 0u;
      } else if (Toggles) {
        // The same packet runs on every outer iteration. An outer step that
        // sets bit 2 means this address has no single bank.
        return std::nullopt;
      }
      Base = Incoming;
      Cursor = nullptr;
      Block = Pred;
      continue;
    }

    const GlobalValue *DefGV = nullptr;
    int64_t Add = 0;
    bool SawImm = false;
    Register Next;
    for (const MachineOperand &MO : Def->operands()) {
      if (MO.isGlobal()) {
        DefGV = MO.getGlobal();
        Add += MO.getOffset();
      } else if (MO.isImm()) {
        Add += MO.getImm();
        SawImm = true;
      } else if (MO.isReg() && MO.isUse() && !Next)
        Next = MO.getReg();
    }
    const unsigned Opc = Def->getOpcode();
    const bool IsAdd = Opc == RISCV::ADDI || Opc == RISCV::C_ADDI ||
                       Opc == RISCV::C_ADDI16SP || Opc == RISCV::C_ADDI4SPN;
    if (DefGV) {
      Disp += Add;
      std::optional<unsigned> Bit = bankFromSection(*DefGV, Disp, Bank1Skew);
      if (!Bit)
        return std::nullopt;
      return withBit(*Bit);
    }
    if (!IsAdd || !Next || !SawImm)
      return std::nullopt;
    Disp += Add;
    Base = Next;
    Cursor = Def;
    Block = Def->getParent();
  }
  return std::nullopt;
}

bool RISCVVLIW::memoryBanksAllow(
    ArrayRef<MachineInstr *> Members, const MachineInstr &Cand,
    function_ref<std::optional<MemBank>(const MachineInstr &)> BankOf) {
  std::optional<MemBank> Existing;
  bool ExistingUnknown = false;
  unsigned Count = 0;
  for (const MachineInstr *MI : Members) {
    if (!MI->mayLoad() && !MI->mayStore())
      continue;
    ++Count;
    std::optional<MemBank> Bank = BankOf(*MI);
    if (!Bank)
      ExistingUnknown = true;
    else
      Existing = Bank;
  }
  if (Count >= 2)
    return false;
  if (Count == 0)
    return true;
  std::optional<MemBank> CandBank = BankOf(Cand);
  if (ExistingUnknown || !CandBank || !Existing)
    return false;
  return banksDifferEveryIteration(*Existing, *CandBank);
}

bool RISCVVLIW::memoryBanksAllow(ArrayRef<MachineInstr *> Members,
                                 const MachineInstr &Cand, unsigned Bank1Skew) {
  return memoryBanksAllow(Members, Cand, [&](const MachineInstr &MI) {
    return memoryBank(MI, Bank1Skew);
  });
}
