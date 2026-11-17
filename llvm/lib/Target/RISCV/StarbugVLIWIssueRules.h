//===- StarbugVLIWIssueRules.h - Lane and bank rules for Starbug ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Issue rules shared by the pre-RA VLIW scheduler and the packetizer.
// Lane legality lives in StarbugVLIWConfig. The bank proof is address bit 2,
// which the schedule model cannot see.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_STARBUGVLIWISSUERULES_H
#define LLVM_LIB_TARGET_RISCV_STARBUGVLIWISSUERULES_H

#include "StarbugVLIWConfig.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <optional>

namespace llvm {

class MachineInstr;

namespace RISCVVLIW {

// Bank is address bit 2. Bit is the bank on iteration 0. Flip is 1 when the
// loop step toggles the bank every iteration. Two formulas differ on every
// iteration when their flips match and their iteration-0 bits do not.
struct MemBank {
  unsigned Bit = 0;
  unsigned Flip = 0;
};

OpClass classifyOpClass(const MachineInstr &MI, bool PacketizeFP);

// addi / c.addi that defines and uses the same register with a nonzero
// immediate. The LSU has no post-increment, so this step occupies a lane.
bool isPointerBump(const MachineInstr &MI);

bool banksDifferEveryIteration(MemBank A, MemBank B);

// Unknown when the bank cannot be proved. An unknown bank is the only memory
// operation allowed in its packet.
std::optional<MemBank> memoryBank(const MachineInstr &MI, unsigned Bank1Skew);

// True when Cand, a memory operation, keeps the packet at one access per bank
// and at most two memory operations. Banks must differ on every iteration.
bool memoryBanksAllow(ArrayRef<MachineInstr *> Members,
                      const MachineInstr &Cand, unsigned Bank1Skew);

bool memoryBanksAllow(
    ArrayRef<MachineInstr *> Members, const MachineInstr &Cand,
    function_ref<std::optional<MemBank>(const MachineInstr &)> BankOf);

} // namespace RISCVVLIW

} // namespace llvm

#endif
