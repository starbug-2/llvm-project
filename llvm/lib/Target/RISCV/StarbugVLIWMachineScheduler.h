//===- StarbugVLIWMachineScheduler.h - Starbug pre-RA VLIW scheduler ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pre-register-allocation scheduler for Starbug. It is the generic converging
// VLIW scheduler with Starbug's lane and bank rules, plus a small cost bias
// for pointer bumps and MAC overlap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_STARBUGVLIWMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_RISCV_STARBUGVLIWMACHINESCHEDULER_H

#include "StarbugVLIWConfig.h"
#include "StarbugVLIWIssueRules.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/VLIWMachineScheduler.h"

namespace llvm {

class StarbugVLIWResourceModel : public VLIWResourceModel {
  RISCVVLIW::StarbugVLIWConfig Config;

  struct BankCacheEntry {
    bool Computed = false;
    bool Known = false;
    RISCVVLIW::MemBank Bank;
  };
  DenseMap<const MachineInstr *, BankCacheEntry> BankCache;

  std::optional<RISCVVLIW::MemBank> cachedBank(const MachineInstr &MI);
  bool lanesAllow(const MachineInstr &Cand) const;
  bool banksAllow(const MachineInstr &Cand);

public:
  StarbugVLIWResourceModel(const TargetSubtargetInfo &STI,
                           const TargetSchedModel *SchedModel);

  bool isResourceAvailable(SUnit *SU, bool IsTop) override;
};

class StarbugConvergingVLIWScheduler : public ConvergingVLIWScheduler {
  RISCVVLIW::StarbugVLIWConfig Config;

protected:
  VLIWResourceModel *
  createVLIWResourceModel(const TargetSubtargetInfo &STI,
                          const TargetSchedModel *SchedModel) const override;

  int SchedulingCost(ReadyQueue &Q, SUnit *SU, SchedCandidate &Candidate,
                     RegPressureDelta &Delta, bool verbose) override;

public:
  StarbugConvergingVLIWScheduler();
};

ScheduleDAGInstrs *createStarbugVLIWMachineSched(MachineSchedContext *C);

} // namespace llvm

#endif
