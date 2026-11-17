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
  // 0 means "derive a factor from the loop and the machine". A non-zero
  // value is an explicit override (-starbug-vliw-unroll-factor) and is still
  // clipped by the spill and I-cache guards.
  bool ForceUnroll = false;
  unsigned DefaultUnrollFactor = 0;
  unsigned MaxUnrollFactor = 8;
  // Unrolled body, in instructions. Kept under the pipeliner's MII cap
  // (pipeliner-max-mii is 27; 64 instructions is an MII of 8 on this machine)
  // and small enough to stay in the I-cache.
  unsigned ICacheBudget = 64;
  // Allocatable registers after a 3/4 headroom. Each unrolled copy is charged
  // for the values it produces, because the modulo scheduler overlaps them.
  unsigned IntRegBudget = 18;
  unsigned FPRegBudget = 18;
  unsigned Threshold = 64;
  unsigned PartialThreshold = 64;
  unsigned MaxPercentThresholdBoost = 100;
  unsigned UnrollAndJamInnerLoopThreshold = 60;
  unsigned SCEVExpansionBudget = 256;
  bool PreferUnrollAndJam = false;
  bool SpillPressureGuard = true;
};

// What one loop looks like to the unroll heuristic. Latencies on this machine
// are one cycle (a packet reads at the start and writes at the end), so a
// carried chain's latency is its instruction count.
struct PipelineUnrollShape {
  unsigned BodyInsts = 0;
  unsigned IndependentInsts = 0;
  unsigned CarriedLat = 1;
  unsigned LiveInt = 0;
  unsigned LiveFP = 0;
  bool PipelinerCandidate = false;
};

struct PipelineUnrollBudget {
  unsigned IssueWidth = 8;
  unsigned HardMax = 8;
  unsigned ExplicitFactor = 0;
  unsigned ICacheInsts = 64;
  unsigned IntBudget = 18;
  unsigned FPBudget = 18;
  bool SpillGuard = true;
};

// Unroll a pipelined loop only far enough to cover its loop-carried chain,
// and only while the copies still fit in the register file. The machine
// pipeliner overlaps the next iteration; a larger factor grows the kernel's
// MII and the I-cache footprint instead of exposing more ILP.
unsigned computePipelineUnrollFactor(const PipelineUnrollShape &Shape,
                                    const PipelineUnrollBudget &Budget);

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
