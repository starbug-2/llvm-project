//===- StarbugVLIWBundleLayout.cpp - Fit bundles inside I-cache lines -----===//
//
// Each execute packet is a 4-byte custom-0 header plus the operations that
// follow it. This pass keeps that packet inside one 64-byte line. A packet
// that would cross the line is split into smaller packets that fit. The
// header is not a nop: splitting removes the straddling header and emits
// headers for the pieces that remain.
//
// Nothing in the compiler could see this before: the packetizer runs on
// MachineInstrs, where byte offsets do not exist yet, so hints landed wherever
// the dependence structure happened to put them. Measured on the CMSIS DSP
// suite, 12-16% of every binary's bundles straddled -- and, weighted by how
// often they actually executed, 12.7% of arm_fir_f32's bundle executions were
// being thrown away by the fetch unit. That is a larger effect than every
// scheduling heuristic in the packetizer put together.
//
// This pass runs last, when instruction sizes are final, and repairs the
// layout by re-placing hints. A header is also what makes a hoisted branch
// legal: the packetizer moves that branch to lane 0, and every member still
// runs before the branch takes effect. Dropping or splitting that header
// without putting the branch back would let a taken branch skip the members
// it used to follow. When an unconditional header cannot cover the whole
// packet, the branch is moved back to the end of the group first.
//
// Two things have to hold for the offsets to be real:
//   * the function must be aligned to the line, which the pass arranges;
//   * linker relaxation must be off, since it deletes bytes inside functions
//     after this point. starbug-vliw clears the relax feature, and
//     -mattr=+relax is rejected in the target machine.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "StarbugVLIWConfig.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "starbug-vliw-bundle-layout"
#define STARBUG_VLIW_BUNDLE_LAYOUT_NAME "Starbug VLIW Bundle Layout"

STATISTIC(NumStraddlingBundles,
          "Number of bundles found straddling an I-cache line");
STATISTIC(NumBundlesRepaired,
          "Number of straddling bundles split to fit inside a line");
STATISTIC(NumHintsDropped,
          "Number of hints removed because no group of two or more fit");
STATISTIC(NumSlotsRecovered,
          "Number of issue slots recovered by repacking straddling bundles");

static cl::opt<bool> StarbugVBundleLayout(
    "starbug-vliw-bundle-layout", cl::init(true), cl::Hidden,
    cl::desc("Repack execute packets that would straddle an I-cache line"));

static cl::opt<unsigned> StarbugVICacheLineBytes(
    "starbug-vliw-icache-line-bytes", cl::init(64), cl::Hidden,
    cl::desc("I-cache line size the bundle layout pass fits bundles into; "
             "must match P.ICACHE_LINELENINBITS/8 in the target config"));

static cl::opt<bool> StarbugVAlignBundledFunctions(
    "starbug-vliw-align-bundled-functions", cl::init(true), cl::Hidden,
    cl::desc("Align functions that contain bundles to the I-cache line so "
             "the layout pass can compute real byte offsets"));

static cl::opt<bool> StarbugVPredictCondBranch(
    "starbug-vliw-layout-predict-cond-branch", cl::init(true), cl::Hidden,
    cl::desc("Assume a compare-against-zero branch emits as C.BEQZ/C.BNEZ; "
             "wrong whenever the backend relaxes it back to four bytes"));

static cl::opt<unsigned> StarbugVLayoutVerbose(
    "starbug-vliw-bundle-layout-verbose", cl::init(0), cl::Hidden,
    cl::desc("Trace every bundle the layout pass walks, to stderr"));

namespace {

class StarbugVLIWBundleLayout : public MachineFunctionPass {
public:
  static char ID;

  StarbugVLIWBundleLayout() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return STARBUG_VLIW_BUNDLE_LAYOUT_NAME;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  /// One layout walk. Repairs the first straddling bundle it finds and
  /// returns true so the caller can recompute offsets, which the repair has
  /// just invalidated for everything downstream.
  bool repairFirstStraddle(MachineBasicBlock &MBB, uint64_t &Offset,
                           const RISCVInstrInfo &TII, unsigned HintBytes,
                           unsigned LineBytes, unsigned MaxWidth, bool HasZca);
};

/// A group of consecutive members that the fetch unit will accept as one
/// bundle, or a single member left to run scalar.
struct Group {
  unsigned First;
  unsigned Count;
  bool Hinted;
};

/// Choose bundle boundaries for `Sizes` laid out from `Cursor`, so that every
/// hinted group -- its four-byte header included -- lies wholly inside one line.
///
/// Greedy, and greedy is right here: taking the widest group that fits now
/// never costs more than one issue cycle later, because whatever is left
/// starts at a line boundary with a full line in front of it.
/// Byte size of \p MI as the emitter will actually write it.
///
/// RISCVInstrInfo::getInstSizeInBytes is an *upper* bound: it is what branch
/// relaxation wants, so for the branch pseudos it reports the uncompressed
/// form. The AsmPrinter, though, runs RISCVRVC::compress on every instruction
/// before it hits the streamer, so an unconditional branch leaves codegen as a
/// two-byte C.J. Believing the upper bound makes every offset past the first
/// branch in a function two bytes too large, and the drift accumulates until
/// the straddle test is meaningless -- which is exactly what it did.
static unsigned predictedSize(const MachineInstr &MI, const RISCVInstrInfo &TII,
                              bool HasZca) {
  const unsigned Size = TII.getInstSizeInBytes(MI);
  if (!HasZca || Size != 4)
    return Size;

  switch (MI.getOpcode()) {
  case RISCV::PseudoBR:
    // JAL x0, target -> C.J target.
    return 2;
  case RISCV::PseudoRET:
    // JALR x0, x1, 0 -> C.JR ra.
    return 2;
  case RISCV::BEQ:
  case RISCV::BNE:
    // C.BEQZ / C.BNEZ: compare against x0, source in the 8-register C set.
    // The backend may relax these back to four bytes when the target is more
    // than +-256 bytes away, so this is the one place the model can still be
    // short. It is right far more often than it is wrong.
    if (StarbugVPredictCondBranch && MI.getOperand(1).isReg() &&
        MI.getOperand(1).getReg() == RISCV::X0 &&
        MI.getOperand(0).isReg() &&
        RISCV::GPRCRegClass.contains(MI.getOperand(0).getReg()))
      return 2;
    return 4;
  default:
    return Size;
  }
}

// The packetizer's only reorder hoists a terminator into lane 0. Without a
// header that still covers every member, a taken branch skips the operations
// that used to sit in front of it. Move the terminator back to the end and
// rotate the sense bits so each operation keeps its predicate.
static void unhoistLeadingTerminator(MachineBasicBlock &MBB,
                                     SmallVectorImpl<MachineInstr *> &Members,
                                     SmallVectorImpl<unsigned> &Sizes,
                                     unsigned &Sense) {
  MachineInstr *Head = Members.front();
  MachineInstr *Last = Members.back();
  MBB.splice(std::next(Last->getIterator()), &MBB, Head->getIterator());
  const unsigned N = Members.size();
  unsigned Rotated = 0;
  if (Sense & RISCVSB2::laneFieldBit(0))
    Rotated |= RISCVSB2::laneFieldBit(N - 1);
  for (unsigned I = 1; I < N && I < 8; ++I)
    if (Sense & RISCVSB2::laneFieldBit(I))
      Rotated |= RISCVSB2::laneFieldBit(I - 1);
  Sense = Rotated;
  Members.erase(Members.begin());
  Members.push_back(Head);
  const unsigned Sz = Sizes.front();
  Sizes.erase(Sizes.begin());
  Sizes.push_back(Sz);
}

static SmallVector<Group, 4> planGroups(ArrayRef<unsigned> Sizes,
                                        uint64_t Cursor, unsigned HintBytes,
                                        unsigned LineBytes,
                                        unsigned MaxWidth) {
  SmallVector<Group, 4> Groups;
  unsigned I = 0;
  const unsigned N = Sizes.size();
  while (I < N) {
    const uint64_t LineEnd = alignTo(Cursor + 1, LineBytes);
    const uint64_t Avail = LineEnd - Cursor;

    uint64_t Need = HintBytes;
    unsigned K = 0;
    while (I + K < N && K < MaxWidth && Need + Sizes[I + K] <= Avail) {
      Need += Sizes[I + K];
      ++K;
    }

    if (K >= 2) {
      Groups.push_back({I, K, true});
      Cursor += Need;
      I += K;
      continue;
    }

    // Nothing worth hinting starts here -- either only one member fits before
    // the line ends, or the member itself crosses the boundary and the fetch
    // unit's spill path will handle it. Let it run scalar and try again from
    // the next one; the cursor has moved, so the window has too.
    Groups.push_back({I, 1, false});
    Cursor += Sizes[I];
    ++I;
  }
  return Groups;
}

} // namespace

char StarbugVLIWBundleLayout::ID = 0;

// The enable flag already owns the "starbug-vliw-bundle-layout" option, so
// the legacy pass argument has to be a different string. opt's pass-name
// parser registers that argument as a command-line option.
INITIALIZE_PASS(StarbugVLIWBundleLayout, "starbug-bundle-layout",
                STARBUG_VLIW_BUNDLE_LAYOUT_NAME, false, false)

FunctionPass *llvm::createStarbugVLIWBundleLayoutPass() {
  return new StarbugVLIWBundleLayout();
}

bool StarbugVLIWBundleLayout::repairFirstStraddle(
    MachineBasicBlock &MBB, uint64_t &Offset, const RISCVInstrInfo &TII,
    unsigned HintBytes, unsigned LineBytes, unsigned MaxWidth,
    bool HasZca) {
  for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE; ++MII) {
    MachineInstr &MI = *MII;
    if (MI.getOpcode() != RISCV::STARBUG_BUNDLE_HINT) {
      unsigned Sz = predictedSize(MI, TII, HasZca);
      if (StarbugVLayoutVerbose > 1) {
        errs() << "  0x" << Twine::utohexstr(Offset) << " +" << Sz << " ";
        if (StarbugVLayoutVerbose > 2)
          MI.print(errs(), /*IsStandalone=*/false, false, false, false);
        else
          errs() << TII.getName(MI.getOpcode()) << "\n";
      }
      Offset += Sz;
      continue;
    }

    const unsigned Len = static_cast<unsigned>(MI.getOperand(0).getImm());

    // Collect the members that actually occupy bytes. Debug values and CFI
    // sit between the hint and its members in the MI stream but emit nothing,
    // so they are not members and must not be counted.
    SmallVector<MachineInstr *, 4> Members;
    SmallVector<unsigned, 4> Sizes;
    uint64_t Extent = HintBytes;
    auto Scan = std::next(MII);
    while (Members.size() < Len && Scan != MIE) {
      unsigned Size = predictedSize(*Scan, TII, HasZca);
      if (Size == 0) {
        ++Scan;
        continue;
      }
      Members.push_back(&*Scan);
      Sizes.push_back(Size);
      Extent += Size;
      if (StarbugVLayoutVerbose > 1) {
        errs() << "  M 0x" << Twine::utohexstr(Offset + Extent - Size) << " +"
               << Size << " ";
        if (StarbugVLayoutVerbose > 2)
          Scan->print(errs(), false, false, false, true);
        else
          errs() << TII.getName(Scan->getOpcode()) << "\n";
      }
      ++Scan;
    }

    const unsigned CReg = static_cast<unsigned>(MI.getOperand(1).getImm());
    unsigned Sense = static_cast<unsigned>(MI.getOperand(2).getImm());
    const bool HasPred = CReg != 0;
    const bool HoistedTerminator =
        !HasPred && Members.size() >= 2 &&
        (Members.front()->isBranch() || Members.front()->isReturn());

    if (Members.size() != Len) {
      // A predicated header is the only thing keeping its operation off the
      // untaken path. Leave it in place. An unconditional hint that outruns
      // the block would make the hardware read whatever followed, so drop it.
      if (HasPred) {
        Offset += HintBytes;
        continue;
      }
      if (HoistedTerminator)
        unhoistLeadingTerminator(MBB, Members, Sizes, Sense);
      Offset += HintBytes;
      MI.eraseFromParent();
      ++NumHintsDropped;
      return true;
    }

    const bool Straddles =
        (Offset / LineBytes) != ((Offset + Extent - 1) / LineBytes);
    if (StarbugVLayoutVerbose)
      errs() << "starbug-layout: " << MBB.getParent()->getName() << " off=0x"
             << Twine::utohexstr(Offset) << " len=" << Len
             << " extent=" << Extent << (Straddles ? "  STRADDLE" : "") << "\n";
    if (!Straddles) {
      Offset += Extent;
      // Extent already covers the members, so step past them. Falling through
      // to the plain-instruction arm would add their sizes a second time and
      // every offset from here on would be fiction.
      MII = std::prev(Scan);
      continue;
    }

    ++NumStraddlingBundles;
    // A straddling header cannot keep a hoisted terminator in lane 0: the
    // pieces will not all share that header, and a taken branch would skip
    // the operations that belong in front of it.
    if (HoistedTerminator)
      unhoistLeadingTerminator(MBB, Members, Sizes, Sense);
    SmallVector<Group, 4> Groups =
        planGroups(Sizes, Offset, HintBytes, LineBytes, MaxWidth);

    // Issue cycles before: one for the hint the core still retires as a NOP,
    // plus one per member run scalar. After: one per hinted group, plus one
    // per member left over.
    unsigned Before = 1 + Len;
    unsigned After = 0;
    for (const Group &G : Groups)
      After += G.Hinted ? 1 : G.Count;
    if (After >= Before) {
      // Repacking would not pay for itself. An unconditional hint buys
      // nothing here and costs a fetched NOP, so drop it. A predicated hint
      // stays: dropping it would run the operation on the untaken path.
      if (HasPred) {
        Offset += Extent;
        MII = std::prev(Scan);
        continue;
      }
      Offset += HintBytes;
      MI.eraseFromParent();
      ++NumHintsDropped;
      LLVM_DEBUG(dbgs() << "starbug-layout: dropped unrepairable hint in "
                        << MBB.getName() << '\n');
      return true;
    }

    LLVM_DEBUG({
      dbgs() << "starbug-layout: bundle of " << Len << " at offset " << Offset
             << " straddles a " << LineBytes << "B line; repacking into";
      for (const Group &G : Groups)
        dbgs() << ' ' << (G.Hinted ? G.Count : 0);
      dbgs() << " (" << Before << " -> " << After << " issue cycles)\n";
    });

    const DebugLoc DL = MI.getDebugLoc();
    MI.eraseFromParent();
    for (const Group &G : Groups) {
      if (G.Hinted) {
        MachineInstr *FirstMember = Members[G.First];
        BuildMI(*FirstMember->getParent(), FirstMember->getIterator(), DL,
                TII.get(RISCV::STARBUG_BUNDLE_HINT))
            .addImm(static_cast<int64_t>(G.Count))
            .addImm(CReg)
            .addImm(RISCVSB2::sliceSense(Sense, G.First, G.Count));
        continue;
      }
      // A member left scalar still has to carry its predicate. Without a
      // header the operation executes on the untaken path.
      if (CReg == 0)
        continue;
      for (unsigned P = 0; P < G.Count; ++P) {
        const unsigned Src = G.First + P;
        MachineInstr *Member = Members[Src];
        const unsigned Bit = Sense & RISCVSB2::laneFieldBit(Src);
        BuildMI(*Member->getParent(), Member->getIterator(), DL,
                TII.get(RISCV::STARBUG_BUNDLE_HINT))
            .addImm(1)
            .addImm(CReg)
            .addImm(Bit ? RISCVSB2::laneFieldBit(0) : 0);
      }
    }

    ++NumBundlesRepaired;
    NumSlotsRecovered += Before - After;
    return true;
  }
  return false;
}

bool StarbugVLIWBundleLayout::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasStarbugVLIW() || !StarbugVBundleLayout)
    return false;

  const auto *TII = ST.getInstrInfo();
  if (!TII)
    return false;

  bool HasHints = false;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() == RISCV::STARBUG_BUNDLE_HINT) {
        HasHints = true;
        break;
      }
    }
    if (HasHints)
      break;
  }
  if (!HasHints)
    return false;

  const unsigned LineBytes = std::max(16u, StarbugVICacheLineBytes.getValue());
  // sb2.header is a 32-bit custom-0 instruction.
  const unsigned HintBytes = 4;
  const unsigned MaxWidth =
      std::min<unsigned>(8, RISCVVLIW::StarbugVLIWConfig::fromCommandLine()
                                .maxBundleWidth());

  bool Changed = false;
  if (StarbugVAlignBundledFunctions && MF.getAlignment() < Align(LineBytes)) {
    // Without this the function's start address is whatever the linker picks
    // and every offset below is fiction.
    MF.ensureAlignment(Align(LineBytes));
    Changed = true;
  }

  // Each repair invalidates the offsets after it, so re-walk. Every round
  // either removes a hint or replaces one straddling hint with groups that do
  // not straddle *at their current offsets*; a later round may still move
  // them, so bound the work by the hint count.
  unsigned Budget = 0;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      if (MI.getOpcode() == RISCV::STARBUG_BUNDLE_HINT)
        Budget += 4;

  for (unsigned Round = 0; Round < Budget; ++Round) {
    uint64_t Offset = 0;
    bool Repaired = false;
    for (MachineBasicBlock &MBB : MF) {
      Offset = alignTo(Offset, MBB.getAlignment());
      if (repairFirstStraddle(MBB, Offset, *TII, HintBytes, LineBytes,
                              MaxWidth, /*HasZca=*/true)) {
        Repaired = true;
        break;
      }
    }
    if (!Repaired)
      break;
    Changed = true;
  }

  return Changed;
}
