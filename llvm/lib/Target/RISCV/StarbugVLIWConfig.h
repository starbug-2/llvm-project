#ifndef LLVM_LIB_TARGET_RISCV_STARBUGVLIWCONFIG_H
#define LLVM_LIB_TARGET_RISCV_STARBUGVLIWCONFIG_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace llvm {

namespace RISCVVLIW {

enum class OpClass : uint8_t {
  Any = 0,      // unrecognised: conservatively lane 0 only
  ALUAddSub,
  ALUMulDiv,
  ALUShift,
  ALULogic,     // and/or/xor and their immediate forms
  Load,
  Store,
  Branch,
  Compare,      // slt/sltu and immediate forms
  Move,         // register moves and immediate materialisation
  CSR,
  FPU,          // FP arithmetic and FP<->int moves/converts/compares
  FPUDivSqrt,   // fdiv / fsqrt: legal anywhere, but stalls execute core-wide
  Last
};

struct LaneConfig {
  unsigned LaneId = 0;
  BitVector AllowedClasses;

  LaneConfig();
  LaneConfig(unsigned LaneId, unsigned NumClasses);
};

struct UnrollPolicy {
  bool ForceUnroll = true;
  unsigned DefaultUnrollFactor = 64;
  unsigned MaxUnrollFactor = 512;
  unsigned Threshold = 100000;
  unsigned PartialThreshold = 100000;
  unsigned MaxPercentThresholdBoost = 1000;
  unsigned UnrollAndJamInnerLoopThreshold = 100000;
  unsigned SCEVExpansionBudget = 4096;
  bool PreferUnrollAndJam = true;
  bool SpillPressureGuard = true;
};

struct SchedulerPolicy {
  bool PrioritizePointerBumps = true;
  bool PrioritizeReadyLoads = true;
  bool PreferLSUAnchoredPackets = true;
  bool PreferMACOverlap = true;
  bool AssumeNoMemoryAlias = false;
  unsigned LatencyWeightLoad = 3;
  unsigned LatencyWeightMulDiv = 2;
  unsigned LatencyWeightStore = 1;
  unsigned PacketizerLookAhead = 64;
  unsigned DependencyLookback = 48;
  bool ReserveLane0ForAny = true;
  bool AllowShortPackets = true;
  // Accept a member whose write is read by an earlier member of the same
  // packet. Safe in hardware (all lanes read before any lane writes back), but
  // only while the packet stays in original program order -- see
  // docs/HARDWARE_CONTRACT.md section 4.1.
  bool AllowIntraPacketWAR = true;
  // Ablation switches. Both default to the correct behaviour; they exist so a
  // sweep can measure what each one is worth rather than assert it.
  bool PacketizeFP = true;          // FP ops may occupy worker lanes
  bool DebugInstrsTransparent = true;  // DBG_VALUE does not break a packet
  bool EmitSingleInstructionHints = false;
  bool PacketizePCRelative = false;
};

struct StarbugVLIWConfig {
  unsigned NumLanes = 8;
  // .bank0 and .bank1 both start 8-byte aligned, so bit 2 of each base is
  // clear. 4 moves .bank1 onto the other bank, matching the scoreboard linker.
  unsigned Bank1Skew = 0;
  SmallVector<LaneConfig, 8> Lanes;
  UnrollPolicy Unroll;
  SchedulerPolicy Scheduler;

  static StarbugVLIWConfig getDefault();
  static StarbugVLIWConfig fromCommandLine();

  bool isLaneLegal(unsigned Lane, OpClass Class) const;
  unsigned maxBundleWidth() const { return NumLanes; }
};

} // namespace RISCVVLIW

} // namespace llvm

#endif
