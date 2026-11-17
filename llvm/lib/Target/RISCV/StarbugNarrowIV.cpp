//===-- StarbugNarrowIV.cpp - 32-bit latch for a 64-bit trip count --------===//
//
// On RV32 a 64-bit induction variable becomes a low half, a carry, and an
// extra branch. That splits the loop, so the steady state pays for a compare
// the trip count does not need when it fits in 32 bits.
//
// ScalarEvolution identifies the affine recurrence. When the values the
// recurrence actually computes fit in 32 bits, the latch is narrowed in
// place. When they might not, the loop is versioned: the wide copy is the
// original, and the narrow copy runs only when the start and the bound both
// stay far enough below 2^32 that the step cannot overflow an i32.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "starbug-narrow-iv"

namespace {

// A truncated use of the IV. Op is an add or sub of the IV with a value
// defined outside the loop when the trunc is of that add or sub. Op is null
// when the trunc reads the IV itself.
struct IndexUse {
  TruncInst *Trunc = nullptr;
  BinaryOperator *Op = nullptr;
};

struct NarrowCandidate {
  Loop *L = nullptr;
  PHINode *IV = nullptr;
  BinaryOperator *Inc = nullptr;
  ICmpInst *Cmp = nullptr;
  Value *Start = nullptr;
  Value *Bound = nullptr;
  uint64_t Step = 0;
  ICmpInst::Predicate Pred = ICmpInst::ICMP_ULT;
  bool CmpOnInc = false;
  BasicBlock *Exit = nullptr;
  SmallVector<IndexUse, 4> Indexes;
  bool NeedsVersion = false;
  uint64_t StartLimit = 0;
  uint64_t BoundLimit = 0;
};

static bool fitsUnsigned32(const APInt &V) { return V.getActiveBits() <= 32; }

static bool collectCandidate(Loop *L, ScalarEvolution &SE, NarrowCandidate &C) {
  if (!L->isInnermost() || L->getNumBlocks() != 1)
    return false;
  BasicBlock *H = L->getHeader();
  BasicBlock *Exit = L->getExitBlock();
  auto *BI = dyn_cast<BranchInst>(H->getTerminator());
  if (!BI || !BI->isConditional() || !Exit)
    return false;
  if (!((BI->getSuccessor(0) == H && BI->getSuccessor(1) == Exit) ||
        (BI->getSuccessor(1) == H && BI->getSuccessor(0) == Exit)))
    return false;

  PHINode *IV = nullptr;
  for (PHINode &PN : H->phis()) {
    if (PN.getType() != Type::getInt64Ty(H->getContext()))
      continue;
    if (IV)
      return false;
    IV = &PN;
  }
  if (!IV || IV->getNumIncomingValues() != 2)
    return false;

  const auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IV));
  if (!AR || AR->getLoop() != L || !AR->isAffine())
    return false;
  const auto *StepS = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
  if (!StepS)
    return false;
  const APInt &StepV = StepS->getAPInt();
  if (!StepV.isStrictlyPositive() || !fitsUnsigned32(StepV))
    return false;

  Value *Start = nullptr;
  BinaryOperator *Inc = nullptr;
  for (unsigned I = 0; I < 2; ++I) {
    Value *V = IV->getIncomingValue(I);
    if (L->contains(IV->getIncomingBlock(I)))
      Inc = dyn_cast<BinaryOperator>(V);
    else
      Start = V;
  }
  if (!Start || !Inc || Inc->getParent() != H ||
      Inc->getOpcode() != Instruction::Add)
    return false;
  Value *StepOp = nullptr;
  if (Inc->getOperand(0) == IV)
    StepOp = Inc->getOperand(1);
  else if (Inc->getOperand(1) == IV)
    StepOp = Inc->getOperand(0);
  else
    return false;
  const auto *StepC = dyn_cast<ConstantInt>(StepOp);
  if (!StepC || StepC->getValue() != StepV)
    return false;

  auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition());
  if (!Cmp || Cmp->getParent() != H)
    return false;

  ICmpInst::Predicate Pred = Cmp->getPredicate();
  Value *IVSide = nullptr;
  Value *Bound = nullptr;
  if (Cmp->getOperand(0) == IV || Cmp->getOperand(0) == Inc) {
    IVSide = Cmp->getOperand(0);
    Bound = Cmp->getOperand(1);
  } else if (Cmp->getOperand(1) == IV || Cmp->getOperand(1) == Inc) {
    IVSide = Cmp->getOperand(1);
    Bound = Cmp->getOperand(0);
    Pred = Cmp->getSwappedPredicate();
  } else {
    return false;
  }
  if (!L->isLoopInvariant(Bound))
    return false;
  // The new compare replaces the old one in place, so the branch sense has
  // to stay "true continues". An inverted latch is left for loop rotation.
  if (BI->getSuccessor(0) != H)
    return false;
  if (Pred != ICmpInst::ICMP_ULT && Pred != ICmpInst::ICMP_ULE)
    return false;

  for (User *U : Inc->users()) {
    if (U == IV || U == Cmp)
      continue;
    return false;
  }

  SmallVector<IndexUse, 4> Indexes;
  Type *I32 = Type::getInt32Ty(H->getContext());
  auto IsHeaderTrunc = [&](Value *V) -> TruncInst * {
    auto *T = dyn_cast<TruncInst>(V);
    if (!T || T->getDestTy() != I32 || T->getParent() != H)
      return nullptr;
    return T;
  };
  for (User *U : IV->users()) {
    if (U == Inc || U == Cmp)
      continue;
    if (TruncInst *T = IsHeaderTrunc(U)) {
      Indexes.push_back({T, nullptr});
      continue;
    }
    auto *Op = dyn_cast<BinaryOperator>(U);
    if (!Op || Op->getParent() != H ||
        (Op->getOpcode() != Instruction::Add &&
         Op->getOpcode() != Instruction::Sub) ||
        !Op->hasOneUse())
      return false;
    Value *Other = Op->getOperand(0) == IV ? Op->getOperand(1) : nullptr;
    if (Op->getOperand(1) == IV)
      Other = Op->getOperand(0);
    if (!Other || Other == IV)
      return false;
    if (auto *OI = dyn_cast<Instruction>(Other))
      if (L->contains(OI->getParent()))
        return false;
    TruncInst *T = IsHeaderTrunc(*Op->user_begin());
    if (!T)
      return false;
    Indexes.push_back({T, Op});
  }

  // Every i32 value the narrow latch computes: the phi, and the increment
  // that the compare (or the backedge) evaluates. The exiting increment is
  // the one that can land one step past the last in-range phi.
  const ConstantRange UR = SE.getUnsignedRange(AR);
  const bool RangeOK = !UR.isWrappedSet() && !UR.isFullSet() &&
                       fitsUnsigned32(UR.getUnsignedMin()) &&
                       fitsUnsigned32(UR.getUnsignedMax());
  const APInt IncMax = UR.getUnsignedMax().zext(128) + StepV.zext(128);
  const bool IncOK = RangeOK && fitsUnsigned32(IncMax);

  // start + step <= 2^32-1, and the exiting value bound+step-1 (ULT) or
  // bound+step (ULE) is still an i32.
  const APInt Two32(128, 1);
  const APInt Limit = Two32 << 32;
  const APInt Step128 = StepV.zext(128);
  const APInt StartLimit = Limit - Step128 - 1;
  const APInt BoundLimit =
      Pred == ICmpInst::ICMP_ULE ? Limit - Step128 - 1 : Limit - Step128;

  auto exceeds = [](Value *V, const APInt &Lim) {
    const auto *K = dyn_cast<ConstantInt>(V);
    return K && K->getValue().zext(128).ugt(Lim);
  };
  if (!IncOK && (exceeds(Start, StartLimit) || exceeds(Bound, BoundLimit)))
    return false;

  C.L = L;
  C.IV = IV;
  C.Inc = Inc;
  C.Cmp = Cmp;
  C.Start = Start;
  C.Bound = Bound;
  C.Step = StepV.getZExtValue();
  C.Pred = Pred;
  C.CmpOnInc = IVSide == Inc;
  C.Exit = Exit;
  C.Indexes = std::move(Indexes);
  C.NeedsVersion = !IncOK;
  C.StartLimit = StartLimit.getZExtValue();
  C.BoundLimit = BoundLimit.getZExtValue();
  return true;
}

static void rewriteIndexes(NarrowCandidate &C, PHINode *NewIV, Type *I32) {
  for (IndexUse &IU : C.Indexes) {
    if (!IU.Op) {
      IU.Trunc->replaceAllUsesWith(NewIV);
      continue;
    }
    Value *Other = IU.Op->getOperand(0) == C.IV ? IU.Op->getOperand(1)
                                                : IU.Op->getOperand(0);
    IRBuilder<> AtOp(IU.Op);
    Value *Other32 = AtOp.CreateTrunc(Other, I32);
    Value *Narrow = nullptr;
    if (IU.Op->getOpcode() == Instruction::Add)
      Narrow = AtOp.CreateAdd(NewIV, Other32, IU.Op->getName());
    else if (IU.Op->getOperand(0) == C.IV)
      Narrow = AtOp.CreateSub(NewIV, Other32, IU.Op->getName());
    else
      Narrow = AtOp.CreateSub(Other32, NewIV, IU.Op->getName());
    IU.Trunc->replaceAllUsesWith(Narrow);
  }
}

static bool narrowLoop(NarrowCandidate &C, LoopInfo &LI, DominatorTree &DT,
                      ScalarEvolution &SE) {
  Loop *L = C.L;
  BasicBlock *H = L->getHeader();
  BasicBlock *PH = L->getLoopPreheader();
  if (!PH) {
    PH = InsertPreheaderForLoop(L, &DT, &LI, nullptr, /*PreserveLCSSA=*/false);
    if (!PH)
      return false;
  }
  if (!PH->getSingleSuccessor())
    return false;

  Function *F = H->getParent();
  LLVMContext &Ctx = H->getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  BasicBlock *NarrowPH = PH;
  ValueToValueMapTy VMap;

  if (C.NeedsVersion) {
    // The clone is spliced in front of the preheader. Splitting a non-empty
    // entry first keeps that clone from becoming the function entry, which
    // the guard would then branch to.
    if (PH->isEntryBlock())
      PH = SplitBlock(PH, PH->getTerminator(), &DT, &LI, nullptr,
                      H->getName() + ".ph");
    // Outside uses of the narrow body must be phis, so the wide copy can
    // supply its own incoming value on the shared exit.
    formLCSSA(*L, DT, &LI, &SE);

    SmallVector<BasicBlock *, 8> WideBlocks;
    Loop *Wide = cloneLoopWithPreheader(PH, PH, L, VMap, ".wide", &LI, &DT,
                                        WideBlocks);
    if (!Wide)
      return false;
    remapInstructionsInBlocks(WideBlocks, VMap);
    auto *WideH = cast<BasicBlock>(VMap[H]);
    auto *WidePH = Wide->getLoopPreheader();
    for (PHINode &Phi : C.Exit->phis()) {
      int Idx = Phi.getBasicBlockIndex(H);
      if (Idx < 0)
        continue;
      Value *IncV = Phi.getIncomingValue(Idx);
      Value *WideInc = VMap.lookup(IncV);
      Phi.addIncoming(WideInc ? WideInc : IncV, WideH);
    }

    NarrowPH = SplitBlock(PH, PH->getTerminator(), &DT, &LI, nullptr,
                          H->getName() + ".narrow.ph");
    IRBuilder<> Guard(PH->getTerminator());
    Value *StartOK =
        Guard.CreateICmpULE(C.Start, ConstantInt::get(I64, C.StartLimit));
    Value *BoundOK =
        Guard.CreateICmpULE(C.Bound, ConstantInt::get(I64, C.BoundLimit));
    Value *NarrowOK = nullptr;
    if (match(StartOK, m_One()))
      NarrowOK = BoundOK;
    else if (match(BoundOK, m_One()))
      NarrowOK = StartOK;
    else
      NarrowOK = Guard.CreateAnd(StartOK, BoundOK, "narrow.iv");
    Guard.CreateCondBr(NarrowOK, NarrowPH, WidePH);
    PH->getTerminator()->eraseFromParent();
  }

  IRBuilder<> AtPH(NarrowPH->getTerminator());
  Value *Start32 = AtPH.CreateTrunc(C.Start, I32, C.IV->getName() + ".start");
  Value *Bound32 = AtPH.CreateTrunc(C.Bound, I32, C.Bound->getName() + ".n");

  PHINode *NewIV =
      PHINode::Create(I32, 2, C.IV->getName() + ".n", H->begin());
  IRBuilder<> AtInc(C.Inc);
  Value *NewInc =
      AtInc.CreateAdd(NewIV, ConstantInt::get(I32, C.Step), C.Inc->getName(),
                      /*HasNUW=*/true, /*HasNSW=*/false);
  NewIV->addIncoming(Start32, NarrowPH);
  NewIV->addIncoming(NewInc, H);

  rewriteIndexes(C, NewIV, I32);

  IRBuilder<> AtCmp(C.Cmp);
  Value *CmpLHS = C.CmpOnInc ? NewInc : NewIV;
  Value *NewCmp = C.Pred == ICmpInst::ICMP_ULT
                      ? AtCmp.CreateICmpULT(CmpLHS, Bound32, C.Cmp->getName())
                      : AtCmp.CreateICmpULE(CmpLHS, Bound32, C.Cmp->getName());
  C.Cmp->replaceAllUsesWith(NewCmp);

  IRBuilder<> AtExit(H->getTerminator());
  for (PHINode &Phi : C.Exit->phis()) {
    int Idx = Phi.getBasicBlockIndex(H);
    if (Idx < 0)
      continue;
    Value *IncV = Phi.getIncomingValue(Idx);
    if (IncV == C.IV)
      Phi.setIncomingValue(Idx, AtExit.CreateZExt(NewIV, I64));
    else if (IncV == C.Inc)
      Phi.setIncomingValue(Idx, AtExit.CreateZExt(NewInc, I64));
  }

  for (IndexUse &IU : C.Indexes)
    IU.Trunc->eraseFromParent();
  for (IndexUse &IU : C.Indexes)
    if (IU.Op)
      IU.Op->eraseFromParent();
  C.Cmp->eraseFromParent();
  C.Inc->replaceAllUsesWith(PoisonValue::get(I64));
  C.Inc->eraseFromParent();
  C.IV->eraseFromParent();

  if (C.NeedsVersion)
    DT.recalculate(*F);
  return true;
}

static bool runImpl(Function &F, LoopInfo &LI, DominatorTree &DT,
                    ScalarEvolution &SE) {
  SmallVector<Loop *, 8> Work;
  for (Loop *L : LI)
    for (Loop *Inner : depth_first(L))
      Work.push_back(Inner);

  SmallVector<NarrowCandidate, 4> Candidates;
  for (Loop *L : Work) {
    NarrowCandidate C;
    if (collectCandidate(L, SE, C))
      Candidates.push_back(std::move(C));
  }

  bool Changed = false;
  for (NarrowCandidate &C : Candidates)
    Changed |= narrowLoop(C, LI, DT, SE);
  if (Changed)
    SE.forgetAllLoops();
  return Changed;
}

class StarbugNarrowIV : public FunctionPass {
public:
  static char ID;
  StarbugNarrowIV() : FunctionPass(ID) {
    initializeStarbugNarrowIVPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    return runImpl(F, LI, DT, SE);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  StringRef getPassName() const override {
    return "Starbug narrow 64-bit loop IV";
  }
};

} // namespace

char StarbugNarrowIV::ID = 0;
INITIALIZE_PASS_BEGIN(StarbugNarrowIV, DEBUG_TYPE,
                      "Narrow 64-bit loop IVs that fit in 32 bits", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(StarbugNarrowIV, DEBUG_TYPE,
                    "Narrow 64-bit loop IVs that fit in 32 bits", false, false)

FunctionPass *llvm::createStarbugNarrowIVPass() { return new StarbugNarrowIV(); }

namespace llvm {

PreservedAnalyses StarbugNarrowIVPass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
  if (!TM->getSubtarget<RISCVSubtarget>(F).hasStarbugVLIW() || F.hasOptNone())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  if (!runImpl(F, LI, DT, SE))
    return PreservedAnalyses::all();

  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

} // namespace llvm
