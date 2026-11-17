#include "StarbugVLIWConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

using namespace llvm;
using namespace llvm::RISCVVLIW;

static cl::opt<unsigned> StarbugVLanes(
    "starbug-vliw-lanes", cl::init(8), cl::Hidden,
    cl::desc("Override Starbug VLIW lane count"));

static cl::opt<bool> StarbugVForceUnroll(
    "starbug-vliw-force-unroll", cl::init(false), cl::Hidden,
    cl::desc("Force the Starbug unroll factor even when the heuristic "
             "would not unroll"));

// 0 asks the TTI to derive the factor from the recurrence, the issue width
// and the register file. A positive value is used as the starting factor and
// then clipped by the spill and I-cache guards.
static cl::opt<unsigned> StarbugVUnrollFactor(
    "starbug-vliw-unroll-factor", cl::init(0), cl::Hidden,
    cl::desc("Starbug unroll factor; 0 derives it from the machine"));

static cl::opt<unsigned> StarbugVMaxUnroll(
    "starbug-vliw-max-unroll", cl::init(8), cl::Hidden,
    cl::desc("Maximum Starbug unroll factor (one copy per lane by default)"));

static cl::opt<bool> StarbugVSpillGuard(
    "starbug-vliw-spill-pressure-guard", cl::init(true), cl::Hidden,
    cl::desc("Shrink the Starbug unroll factor when copies would spill"));

static cl::opt<unsigned> StarbugVUnrollThreshold(
    "starbug-vliw-unroll-threshold", cl::init(400), cl::Hidden,
    cl::desc("Loop unroll threshold override for Starbug VLIW"));

static cl::opt<unsigned> StarbugVPartialUnrollThreshold(
    "starbug-vliw-partial-unroll-threshold", cl::init(300), cl::Hidden,
    cl::desc("Partial/runtime loop unroll threshold override for Starbug VLIW"));

static cl::opt<unsigned> StarbugVUnrollMaxPercentThresholdBoost(
    "starbug-vliw-unroll-max-percent-threshold-boost", cl::init(400),
    cl::Hidden,
    cl::desc("Max percent threshold boost for full unroll profitability"));

static cl::opt<unsigned> StarbugVUnrollAndJamInnerThreshold(
    "starbug-vliw-unroll-and-jam-inner-threshold", cl::init(200),
    cl::Hidden,
    cl::desc("Inner-loop threshold for unroll-and-jam profitability"));

static cl::opt<unsigned> StarbugVSCEVExpansionBudget(
    "starbug-vliw-scev-expansion-budget", cl::init(4096), cl::Hidden,
    cl::desc("SCEV expansion budget for runtime unroll analysis"));

static cl::opt<std::string> StarbugVLaneOpClasses(
    "starbug-vliw-lane-op-classes", cl::init(""), cl::Hidden,
    cl::desc("Override per-lane op classes. Format: "
             "lane:class[,class];lane:class[,class]. Example: "
             "0:any;1:alu_addsub,alu_muldiv,alu_shift"));

static cl::opt<bool> StarbugVEmitSingleHints(
    "starbug-vliw-emit-single-hints", cl::init(false), cl::Hidden,
    cl::desc("Emit STARBUG_BUNDLE_HINT for single-instruction packets"));

static cl::opt<bool> StarbugVPacketizePCRel(
    "starbug-vliw-packetize-pcrel", cl::init(false), cl::Hidden,
    cl::desc("Allow packetizing PC-relative address setup instructions"));

static cl::opt<unsigned> StarbugVBank1Skew(
    "starbug-bank1-skew", cl::init(0), cl::Hidden,
    cl::desc("Byte skew of the .bank1 section. 0 and 4 match the linker; "
             "4 flips .bank1 onto the other memory bank"));

static cl::opt<unsigned> StarbugVPacketizerLookAhead(
    "starbug-vliw-packetizer-lookahead", cl::init(64), cl::Hidden,
    cl::desc("Max forward search window for filling Starbug VLIW packets"));

static cl::opt<unsigned> StarbugVDependencyLookback(
    "starbug-vliw-dependency-lookback", cl::init(48), cl::Hidden,
    cl::desc("Backward search window for producer/consumer pairing heuristics"));

static cl::opt<bool> StarbugVReserveLane0(
    "starbug-vliw-reserve-lane0-for-any", cl::init(true), cl::Hidden,
    cl::desc("Prefer non-zero lanes for ALU ops to keep lane 0 open for any-op"));

static cl::opt<bool> StarbugVAllowIntraPacketWAR(
    "starbug-vliw-allow-intra-packet-war", cl::init(true), cl::Hidden,
    cl::desc("Allow an anti-dependence inside a bundle when the bundle keeps "
             "its original instruction order"));

static cl::opt<bool> StarbugVPacketizeFP(
    "starbug-vliw-packetize-fp", cl::init(true), cl::Hidden,
    cl::desc("Allow floating-point operations in worker lanes (ablation: "
             "false reproduces the lane-0-only-FP behaviour)"));

static cl::opt<bool> StarbugVDebugTransparent(
    "starbug-vliw-debug-instrs-transparent", cl::init(true), cl::Hidden,
    cl::desc("Step over debug instructions instead of ending the packet at "
             "them (ablation: false reproduces -g suppressing bundling)"));

static cl::opt<bool> StarbugVPreferLSUAnchoredPackets(
    "starbug-vliw-prefer-lsu-anchored-packets", cl::init(true), cl::Hidden,
    cl::desc("Prefer LSU operations as packet anchors to overlap memory and compute"));

static cl::opt<bool> StarbugVPreferMACOverlap(
    "starbug-vliw-prefer-mac-overlap", cl::init(true), cl::Hidden,
    cl::desc("Prefer mul/add/store producer-consumer chains for MAC-style overlap"));

static cl::opt<bool> StarbugVAssumeNoMemoryAlias(
    "starbug-vliw-assume-no-memory-alias", cl::init(false), cl::Hidden,
    cl::desc("Allow load/store reordering in packetizer assuming input/output do not alias"));

static cl::opt<bool> StarbugVAssumeDisjointMemory(
    "starbug-vliw-assume-disjoint-memory", cl::init(false), cl::Hidden,
    cl::desc("Allow load/store reordering in packetizer when stream bases are disjoint"));

LaneConfig::LaneConfig() = default;

LaneConfig::LaneConfig(unsigned LaneId, unsigned NumClasses)
    : LaneId(LaneId), AllowedClasses(NumClasses, false) {}

static LaneConfig makeLane0() {
  LaneConfig L(0, static_cast<unsigned>(OpClass::Last));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::Any));
  return L;
}

// Worker lanes are full IEU instances with only their memory, branch and CSR
// paths disconnected, so every purely register-to-register integer operation is
// legal on them -- including logic, compare and move, which earlier versions
// omitted. Leaving those out confined kernels built from xor/and/or (hashing,
// crypto, bit manipulation) to lane 0 and capped their bundles at two.
//
// The same applies to floating point, which earlier versions also omitted.
// wallypipelinedcore.sv instantiates fpu_1/fpu_2/fpu_3 next to the lane IEUs,
// backed by fregfile_widened (four write ports, three read ports per lane), and
// routes each lane's FP-to-integer results (FIntResM_n, FCvtIntResW_n,
// FIntDivResultW_n) back into that lane's IEU writeback. Excluding FP confined
// every DSP kernel -- the workload STARBUG exists for -- to bundling its
// address arithmetic while the fmadd chain that dominates the loop ran scalar.
static LaneConfig makeALULane(unsigned LaneId) {
  LaneConfig L(LaneId, static_cast<unsigned>(OpClass::Last));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALUAddSub));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALUMulDiv));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALUShift));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::ALULogic));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::Compare));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::Move));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::FPU));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::FPUDivSqrt));
  // Two memory operations may issue together, on any lanes, when they touch
  // different banks. The packetizer checks the bank. The lane only has to
  // accept the opcode.
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::Load));
  L.AllowedClasses.set(static_cast<unsigned>(OpClass::Store));
  return L;
}

static bool applyClassToken(StringRef Token, BitVector &Allowed) {
  Token = Token.trim();
  if (Token.empty())
    return true;
  const std::string Lower = Token.lower();
  const StringRef Key = Lower;

  auto setClass = [&](OpClass C) {
    Allowed.set(static_cast<unsigned>(C));
  };

  if (Key == "any" || Key == "all")
    setClass(OpClass::Any);
  else if (Key == "alu" || Key == "int") {
    setClass(OpClass::ALUAddSub);
    setClass(OpClass::ALUMulDiv);
    setClass(OpClass::ALUShift);
    setClass(OpClass::ALULogic);
    setClass(OpClass::Compare);
    setClass(OpClass::Move);
  } else if (Key == "alu_logic" || Key == "logic")
    setClass(OpClass::ALULogic);
  else if (Key == "alu_addsub" || Key == "addsub" || Key == "add" ||
             Key == "sub")
    setClass(OpClass::ALUAddSub);
  else if (Key == "alu_muldiv" || Key == "muldiv" || Key == "mul" ||
           Key == "div")
    setClass(OpClass::ALUMulDiv);
  else if (Key == "alu_shift" || Key == "shift" || Key == "sh")
    setClass(OpClass::ALUShift);
  else if (Key == "load" || Key == "ld")
    setClass(OpClass::Load);
  else if (Key == "store" || Key == "st")
    setClass(OpClass::Store);
  else if (Key == "branch" || Key == "br")
    setClass(OpClass::Branch);
  else if (Key == "compare" || Key == "cmp")
    setClass(OpClass::Compare);
  else if (Key == "move" || Key == "mv" || Key == "li" ||
           Key == "copy")
    setClass(OpClass::Move);
  else if (Key == "csr")
    setClass(OpClass::CSR);
  else if (Key == "fp" || Key == "fpu" || Key == "float") {
    setClass(OpClass::FPU);
    setClass(OpClass::FPUDivSqrt);
  } else if (Key == "fp_arith" || Key == "fpu_arith")
    setClass(OpClass::FPU);
  else if (Key == "fp_divsqrt" || Key == "fdiv" || Key == "fsqrt")
    setClass(OpClass::FPUDivSqrt);
  else
    return false;

  return true;
}

static bool parseLaneClassList(StringRef LaneClasses, LaneConfig &Lane) {
  Lane.AllowedClasses.reset();

  SmallVector<StringRef, 16> ClassTokens;
  LaneClasses.split(ClassTokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Token : ClassTokens) {
    if (!applyClassToken(Token, Lane.AllowedClasses))
      return false;
  }

  return true;
}

static void ensureLaneExists(StarbugVLIWConfig &Cfg, unsigned LaneId) {
  for (unsigned I = Cfg.Lanes.size(); I <= LaneId; ++I) {
    if (I == 0)
      Cfg.Lanes.push_back(makeLane0());
    else
      Cfg.Lanes.push_back(makeALULane(I));
  }
}

static bool applyLaneOpClassOverrides(StarbugVLIWConfig &Cfg, StringRef Spec) {
  Spec = Spec.trim();
  if (Spec.empty())
    return true;

  SmallVector<StringRef, 8> LaneSpecs;
  Spec.split(LaneSpecs, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef LaneSpec : LaneSpecs) {
    LaneSpec = LaneSpec.trim();
    if (LaneSpec.empty())
      continue;

    auto Pair = LaneSpec.split(':');
    StringRef LaneIdStr = Pair.first.trim();
    StringRef ClassesStr = Pair.second.trim();
    if (LaneIdStr.empty() || ClassesStr.empty())
      return false;

    unsigned LaneId = 0;
    if (LaneIdStr.getAsInteger(10, LaneId))
      return false;

    ensureLaneExists(Cfg, LaneId);
    Cfg.Lanes[LaneId].LaneId = LaneId;
    if (!parseLaneClassList(ClassesStr, Cfg.Lanes[LaneId]))
      return false;
  }

  Cfg.NumLanes = std::max(Cfg.NumLanes, static_cast<unsigned>(Cfg.Lanes.size()));
  return true;
}

StarbugVLIWConfig StarbugVLIWConfig::getDefault() {
  StarbugVLIWConfig Cfg;
  Cfg.NumLanes = 8;

  Cfg.Lanes.clear();
  Cfg.Lanes.push_back(makeLane0());
  for (unsigned Lane = 1; Lane < Cfg.NumLanes; ++Lane)
    Cfg.Lanes.push_back(makeALULane(Lane));

  Cfg.Unroll.ForceUnroll = false;
  Cfg.Unroll.DefaultUnrollFactor = 0;
  Cfg.Unroll.MaxUnrollFactor = 8;
  Cfg.Unroll.ICacheBudget = 64;
  Cfg.Unroll.IntRegBudget = 18;
  Cfg.Unroll.FPRegBudget = 18;
  Cfg.Unroll.Threshold = 64;
  Cfg.Unroll.PartialThreshold = 64;
  Cfg.Unroll.MaxPercentThresholdBoost = 100;
  Cfg.Unroll.UnrollAndJamInnerLoopThreshold = 60;
  Cfg.Unroll.SCEVExpansionBudget = 256;
  Cfg.Unroll.PreferUnrollAndJam = false;
  Cfg.Unroll.SpillPressureGuard = true;

  Cfg.Scheduler.PrioritizePointerBumps = true;
  Cfg.Scheduler.PrioritizeReadyLoads = true;
  Cfg.Scheduler.PreferLSUAnchoredPackets = true;
  Cfg.Scheduler.PreferMACOverlap = true;
  Cfg.Scheduler.AssumeNoMemoryAlias = false;
  Cfg.Scheduler.PacketizerLookAhead = 64;
  Cfg.Scheduler.DependencyLookback = 48;
  Cfg.Scheduler.ReserveLane0ForAny = true;
  Cfg.Scheduler.AllowShortPackets = true;
  Cfg.Scheduler.AllowIntraPacketWAR = true;
  Cfg.Scheduler.PacketizeFP = true;
  Cfg.Scheduler.DebugInstrsTransparent = true;
  Cfg.Scheduler.EmitSingleInstructionHints = false;
  Cfg.Scheduler.PacketizePCRelative = false;

  return Cfg;
}

StarbugVLIWConfig StarbugVLIWConfig::fromCommandLine() {
  StarbugVLIWConfig Cfg = getDefault();

  Cfg.NumLanes = std::max(1u, StarbugVLanes.getValue());
  Cfg.Bank1Skew = StarbugVBank1Skew.getValue();
  Cfg.Unroll.ForceUnroll = StarbugVForceUnroll;
  Cfg.Unroll.DefaultUnrollFactor = std::min(512u, StarbugVUnrollFactor.getValue());
  Cfg.Unroll.MaxUnrollFactor = std::min(512u, StarbugVMaxUnroll.getValue());
  Cfg.Unroll.Threshold = StarbugVUnrollThreshold;
  Cfg.Unroll.PartialThreshold = StarbugVPartialUnrollThreshold;
  Cfg.Unroll.MaxPercentThresholdBoost =
      std::max(100u, StarbugVUnrollMaxPercentThresholdBoost.getValue());
  Cfg.Unroll.UnrollAndJamInnerLoopThreshold =
      StarbugVUnrollAndJamInnerThreshold;
  Cfg.Unroll.SCEVExpansionBudget =
      std::max(1u, StarbugVSCEVExpansionBudget.getValue());
  Cfg.Unroll.SpillPressureGuard = StarbugVSpillGuard;
  if (Cfg.Unroll.MaxUnrollFactor == 0)
    Cfg.Unroll.MaxUnrollFactor = 1;
  // An explicit factor is allowed to raise the ceiling so
  // -starbug-vliw-unroll-factor=N is enough for a sweep. The spill guard
  // still shrinks N when the copies would not fit.
  if (Cfg.Unroll.DefaultUnrollFactor)
    Cfg.Unroll.MaxUnrollFactor = std::max(Cfg.Unroll.MaxUnrollFactor,
                                         Cfg.Unroll.DefaultUnrollFactor);
  Cfg.Scheduler.PacketizerLookAhead = std::max(1u, StarbugVPacketizerLookAhead.getValue());
  Cfg.Scheduler.DependencyLookback = std::max(1u, StarbugVDependencyLookback.getValue());
  Cfg.Scheduler.ReserveLane0ForAny = StarbugVReserveLane0;
  Cfg.Scheduler.PreferLSUAnchoredPackets = StarbugVPreferLSUAnchoredPackets;
  Cfg.Scheduler.PreferMACOverlap = StarbugVPreferMACOverlap;
  Cfg.Scheduler.AssumeNoMemoryAlias =
      StarbugVAssumeNoMemoryAlias || StarbugVAssumeDisjointMemory;
  Cfg.Scheduler.AllowIntraPacketWAR = StarbugVAllowIntraPacketWAR;
  Cfg.Scheduler.PacketizeFP = StarbugVPacketizeFP;
  Cfg.Scheduler.DebugInstrsTransparent = StarbugVDebugTransparent;
  Cfg.Scheduler.EmitSingleInstructionHints = StarbugVEmitSingleHints;
  Cfg.Scheduler.PacketizePCRelative = StarbugVPacketizePCRel;

  // TODO: Replace this with generated config loading from YAML-derived data.
  // For now, expand lane vector using lane0-any + ALU-only lanes.
  Cfg.Lanes.clear();
  Cfg.Lanes.push_back(makeLane0());
  for (unsigned Lane = 1; Lane < Cfg.NumLanes; ++Lane)
    Cfg.Lanes.push_back(makeALULane(Lane));

  if (!applyLaneOpClassOverrides(Cfg, StarbugVLaneOpClasses)) {
    errs() << "warning: ignoring invalid -starbug-vliw-lane-op-classes spec: "
           << StarbugVLaneOpClasses << '\n';
  }

  return Cfg;
}

static unsigned floorPow2(unsigned N) {
  if (N <= 1)
    return 1;
  return 1u << (31 - llvm::countl_zero(N));
}

unsigned RISCVVLIW::computePipelineUnrollFactor(
    const PipelineUnrollShape &Shape, const PipelineUnrollBudget &Budget) {
  unsigned UF = 1;
  if (Budget.ExplicitFactor) {
    UF = Budget.ExplicitFactor;
  } else if (!Shape.PipelinerCandidate) {
    // No modulo schedule will overlap iterations. Unroll until one copy of
    // the body can fill the issue width, and no further.
    const unsigned Indep = std::max(1u, Shape.IndependentInsts);
    UF = std::max(1u, (Budget.IssueWidth + Indep - 1) / Indep);
    UF = floorPow2(UF);
  } else {
    // The pipeliner hides a carried chain by overlapping the next iteration.
    // Extra copies only help when the chain is longer than the independent
    // work already in the body; they do not create ILP on their own.
    // Accumulator splitting is what breaks a reduction into independent chains.
    const unsigned Indep = Shape.IndependentInsts;
    if (Indep == 0)
      UF = std::max(1u, Shape.CarriedLat);
    else
      UF = std::max(1u, (Shape.CarriedLat + Indep - 1) / Indep);
    UF = floorPow2(UF);
  }

  UF = std::min(UF, std::max(1u, Budget.HardMax));
  if (!Budget.SpillGuard)
    return std::max(1u, UF);

  auto fits = [&](unsigned Factor) {
    if (Shape.LiveInt && Factor * Shape.LiveInt > Budget.IntBudget)
      return false;
    if (Shape.LiveFP && Factor * Shape.LiveFP > Budget.FPBudget)
      return false;
    if (Shape.BodyInsts && Factor * Shape.BodyInsts > Budget.ICacheInsts)
      return false;
    return true;
  };
  while (UF > 1 && !fits(UF))
    UF >>= 1;
  return std::max(1u, UF);
}

bool StarbugVLIWConfig::isLaneLegal(unsigned Lane, OpClass Class) const {
  if (Lane >= Lanes.size())
    return false;

  const auto &Allowed = Lanes[Lane].AllowedClasses;
  const unsigned AnyClass = static_cast<unsigned>(OpClass::Any);
  const unsigned RequestedClass = static_cast<unsigned>(Class);

  if (AnyClass < Allowed.size() && Allowed.test(AnyClass))
    return true;

  return RequestedClass < Allowed.size() && Allowed.test(RequestedClass);
}
