//===-- StarbugSplitAccum.cpp - independent reduction chains --------------===//
//
// A single accumulator is a loop-carried chain. On an 8-wide packet that chain
// issues once per recurrence, no matter how many other lanes are free. The
// machine pipeliner overlaps iterations; it cannot overlap a dependence on
// the value it just produced. Splitting the reduction into 4 or 8 partial
// sums, one per chain, is what makes those lanes useful. The partials are
// folded once after the loop.
//
// Integer add/mul/and/or/xor reassociate modulo 2^n, so wrapping operators
// are always split. nsw/nuw operators are left alone unless
// -starbug-reassoc-reductions is set, because a partial sum can overflow
// when the total does not. Floating-point reductions are split only when the
// operator allows reassociation (-ffast-math sets that flag) or the same
// explicit flag is set.
//
// The transform itself unrolls the body by the chain count and then marks
// the loop so the generic unroller does not multiply it again. Further
// overlap is the pipeliner's job.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "StarbugVLIWConfig.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

#define DEBUG_TYPE "starbug-split-accum"

static cl::opt<bool> StarbugSplitAccumEnable(
    "starbug-split-accum", cl::init(true), cl::Hidden,
    cl::desc("Split Starbug reductions into independent accumulator chains"));

static cl::opt<bool> StarbugReassocReductions(
    "starbug-reassoc-reductions", cl::init(false), cl::Hidden,
    cl::desc("Reassociate reductions even without fast-math flags or nsw"));

namespace {

struct SplitPlan {
  Loop *L = nullptr;
  PHINode *IV = nullptr;
  BinaryOperator *Inc = nullptr;
  PHINode *RedPhi = nullptr;
  Instruction *RedOp = nullptr;
  RecurKind Kind = RecurKind::None;
  ICmpInst *Cmp = nullptr;
  unsigned Chains = 0;
  unsigned ChainLog = 0;
  SmallVector<Instruction *, 8> Work;
};

static bool isEmptyBlock(const BasicBlock *BB) {
  for (const Instruction &I : *BB)
    if (!I.isTerminator() && !I.isDebugOrPseudoInst())
      return false;
  return true;
}

static bool allowsReassoc(Instruction *RedOp, RecurKind Kind) {
  switch (Kind) {
  case RecurKind::And:
  case RecurKind::Or:
  case RecurKind::Xor:
    return true;
  case RecurKind::Add:
  case RecurKind::Mul: {
    auto *BO = dyn_cast<BinaryOperator>(RedOp);
    if (!BO)
      return false;
    if (!BO->hasNoSignedWrap() && !BO->hasNoUnsignedWrap())
      return true;
    return StarbugReassocReductions;
  }
  case RecurKind::FAdd:
  case RecurKind::FMul:
  case RecurKind::FMulAdd:
    if (StarbugReassocReductions)
      return true;
    return RedOp->getFastMathFlags().allowReassoc();
  default:
    return false;
  }
}

static unsigned termRegs(ArrayRef<Instruction *> Work) {
  unsigned N = 0;
  for (Instruction *I : Work) {
    if (I->getType()->isVoidTy() || isa<GetElementPtrInst>(I))
      continue;
    if (I->getType()->isIntegerTy() || I->getType()->isFloatingPointTy())
      ++N;
  }
  return N;
}

// Four chains when eight would not fit in the integer/FP file. Zero when
// even four partials would spill: a spill is another memory op and wipes
// out the ILP the split was for.
static unsigned pickChains(unsigned TermResults, unsigned Lanes) {
  const unsigned Budget = 18;
  const unsigned Per = TermResults + 1;
  unsigned Chains = Lanes >= 8 ? 8 : 4;
  if (Chains == 8 && Chains * Per > Budget)
    Chains = 4;
  if (Chains * Per > Budget)
    return 0;
  return Chains;
}

static bool collectPlan(Loop *L, ScalarEvolution &SE, unsigned Lanes,
                        SplitPlan &P) {
  if (hasDisableAllTransformsHint(L) || !L->isInnermost() ||
      L->getNumBlocks() != 1)
    return false;
  BasicBlock *H = L->getHeader();
  BasicBlock *Exit = L->getExitBlock();
  BasicBlock *PH = L->getLoopPreheader();
  if (!H || !Exit || !PH || !isEmptyBlock(PH))
    return false;
  auto *BI = dyn_cast<BranchInst>(H->getTerminator());
  if (!BI || !BI->isConditional())
    return false;
  auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition());
  if (!Cmp || !Cmp->hasOneUse())
    return false;

  PHINode *RedPhi = nullptr;
  RecurrenceDescriptor RD;
  for (PHINode &PN : H->phis()) {
    RecurrenceDescriptor One;
    if (!RecurrenceDescriptor::isReductionPHI(&PN, L, One, nullptr, nullptr,
                                              nullptr, &SE))
      continue;
    if (RedPhi)
      return false;
    RedPhi = &PN;
    RD = One;
  }
  if (!RedPhi || RedPhi->getNumIncomingValues() != 2)
    return false;

  Instruction *RedOp =
      dyn_cast<Instruction>(RedPhi->getIncomingValueForBlock(H));
  if (!RedOp || RedOp->getParent() != H)
    return false;
  // FindLast, AnyOf, and FindIV are recurrences, but getReductionOpChain asks
  // for a single opcode and aborts on those kinds. They are not reassociable
  // sums, so reject them before that walk.
  if (!allowsReassoc(RedOp, RD.getRecurrenceKind()))
    return false;
  auto Chain = RD.getReductionOpChain(RedPhi, L);
  if (Chain.size() != 1 || Chain[0] != RedOp)
    return false;
  if (RD.hasUsesOutsideReductionChain())
    return false;

  PHINode *IV = nullptr;
  for (PHINode &PN : H->phis()) {
    if (&PN == RedPhi)
      continue;
    if (IV)
      return false;
    IV = &PN;
  }
  if (!IV || !IV->getType()->isIntegerTy() || IV->getNumIncomingValues() != 2)
    return false;
  const auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IV));
  if (!AR || AR->getLoop() != L || !AR->isAffine())
    return false;
  const auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
  if (!Step || !Step->getAPInt().isOne())
    return false;

  Value *Latch = IV->getIncomingValueForBlock(H);
  auto *Inc = dyn_cast<BinaryOperator>(Latch);
  if (!Inc || Inc->getOpcode() != Instruction::Add || Inc->getParent() != H)
    return false;
  ConstantInt *StepC = nullptr;
  if (Inc->getOperand(0) == IV)
    StepC = dyn_cast<ConstantInt>(Inc->getOperand(1));
  else if (Inc->getOperand(1) == IV)
    StepC = dyn_cast<ConstantInt>(Inc->getOperand(0));
  if (!StepC || !StepC->isOne())
    return false;
  for (User *U : Inc->users())
    if (U != IV && U != Cmp)
      return false;

  const SCEV *BTC = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BTC))
    return false;
  // The single-argument form widens the count by one bit so the +1 cannot
  // wrap. The latch arithmetic is in the IV type, so ask for that type.
  const SCEV *Trip = SE.getTripCountFromExitCount(BTC, IV->getType(), L);
  if (isa<SCEVCouldNotCompute>(Trip) || Trip->getType() != IV->getType())
    return false;
  if (const auto *TC = dyn_cast<SCEVConstant>(Trip)) {
    if (TC->getAPInt().ult(4))
      return false;
  }

  for (PHINode &Phi : Exit->phis()) {
    Value *IncV = Phi.getIncomingValueForBlock(H);
    if (IncV != RedOp && IncV != RedPhi)
      return false;
  }
  for (User *U : RedOp->users()) {
    auto *UI = dyn_cast<Instruction>(U);
    if (!UI)
      return false;
    if (!L->contains(UI) && UI->getParent() != Exit)
      return false;
  }
  for (User *U : RedPhi->users()) {
    auto *UI = dyn_cast<Instruction>(U);
    if (!UI || (!L->contains(UI) && UI->getParent() != Exit))
      return false;
  }

  SmallVector<Instruction *, 8> Work;
  for (Instruction &I : *H) {
    if (&I == IV || &I == RedPhi || &I == RedOp || &I == Inc || &I == Cmp ||
        &I == BI || I.isDebugOrPseudoInst())
      continue;
    if (I.mayWriteToMemory() || isa<CallBase>(I))
      return false;
    for (Value *Op : I.operands())
      if (Op == RedPhi || Op == RedOp)
        return false;
    Work.push_back(&I);
  }

  const unsigned Chains = pickChains(termRegs(Work), Lanes);
  if (Chains < 4)
    return false;

  P.L = L;
  P.IV = IV;
  P.Inc = Inc;
  P.RedPhi = RedPhi;
  P.RedOp = RedOp;
  P.Kind = RD.getRecurrenceKind();
  P.Cmp = Cmp;
  P.Chains = Chains;
  P.ChainLog = llvm::countr_zero(Chains);
  P.Work = std::move(Work);
  return true;
}

static void dropWrap(Instruction *I) {
  // fadd and fmul are binary operators too, and they have no wrap flags.
  if (isa<OverflowingBinaryOperator>(I)) {
    I->setHasNoUnsignedWrap(false);
    I->setHasNoSignedWrap(false);
  }
}

static Value *combine(IRBuilder<> &B, Value *Acc, Value *Part, unsigned Opcode,
                      Instruction *FlagsFrom) {
  Value *R = B.CreateBinOp(static_cast<Instruction::BinaryOps>(Opcode), Acc,
                           Part, "rdx.fold");
  if (auto *I = dyn_cast<Instruction>(R))
    I->copyIRFlags(FlagsFrom, /*IncludeWrapFlags=*/false);
  return R;
}

static bool splitLoop(SplitPlan &P, LoopInfo &LI, DominatorTree &DT,
                      ScalarEvolution &SE) {
  Loop *L = P.L;
  BasicBlock *H = L->getHeader();
  BasicBlock *Exit = L->getExitBlock();
  BasicBlock *PH = L->getLoopPreheader();
  Function *F = H->getParent();
  LLVMContext &Ctx = H->getContext();
  Type *IVTy = P.IV->getType();
  auto *BI = cast<BranchInst>(H->getTerminator());

  // Stay in the IV type. The widening overload returns an i(N+1) count, which
  // the expander will not cast down for us.
  const SCEV *TripS = SE.getTripCountFromExitCount(
      SE.getBackedgeTakenCount(L), IVTy, L);
  SCEVExpander Exp(SE, "starbug.split", /*PreserveLCSSA=*/false);
  if (isa<SCEVCouldNotCompute>(TripS) || !Exp.isSafeToExpand(TripS))
    return false;

  Value *Start = nullptr;
  for (unsigned I = 0; I < 2; ++I)
    if (!L->contains(P.IV->getIncomingBlock(I)))
      Start = P.IV->getIncomingValue(I);
  Value *OrigInit = nullptr;
  for (unsigned I = 0; I < 2; ++I)
    if (!L->contains(P.RedPhi->getIncomingBlock(I)))
      OrigInit = P.RedPhi->getIncomingValue(I);
  if (!Start || !OrigInit)
    return false;

  PHINode *Result = nullptr;
  for (PHINode &Phi : Exit->phis())
    if (Phi.getIncomingValueForBlock(H) == P.RedOp ||
        Phi.getIncomingValueForBlock(H) == P.RedPhi)
      Result = &Phi;

  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> RemBlocks;
  // cloneLoopWithPreheader splices the copy in front of the preheader. The
  // function entry cannot be that block: the copy would become the entry and
  // the guard edge would then be a predecessor of the entry.
  if (PH->isEntryBlock())
    PH = SplitBlock(PH, PH->getTerminator(), &DT, &LI, nullptr, "split.pre");
  cloneLoopWithPreheader(PH, PH, L, VMap, ".rem", &LI, &DT, RemBlocks);
  remapInstructionsInBlocks(RemBlocks, VMap);
  auto *RemH = cast<BasicBlock>(VMap[H]);
  auto *RemPH = cast<BasicBlock>(VMap[PH]);
  auto *RemIV = cast<PHINode>(VMap[P.IV]);
  auto *RemRed = cast<PHINode>(VMap[P.RedPhi]);
  Value *RemExitVal = VMap.lookup(P.RedOp);
  if (!RemExitVal)
    RemExitVal = VMap.lookup(P.RedPhi);

  Value *TripV = Exp.expandCodeFor(TripS, IVTy, PH->getTerminator()->getIterator());
  Exp.clear();

  BasicBlock *NarrowPH =
      SplitBlock(PH, PH->getTerminator(), &DT, &LI, nullptr, "split.ph");

  BasicBlock *MainExit = BasicBlock::Create(Ctx, "main.exit", F, Exit);
  BasicBlock *RemGuard = BasicBlock::Create(Ctx, "rem.guard", F, Exit);
  BasicBlock *RemExit = BasicBlock::Create(Ctx, "rem.exit", F, Exit);
  if (Loop *Parent = L->getParentLoop()) {
    Parent->addBasicBlockToLoop(MainExit, LI);
    Parent->addBasicBlockToLoop(RemGuard, LI);
    Parent->addBasicBlockToLoop(RemExit, LI);
  }

  IRBuilder<> Guard(PH->getTerminator());
  Value *Mask = ConstantInt::get(IVTy, P.Chains - 1);
  Value *Rem = Guard.CreateAnd(TripV, Mask, "rdx.rem");
  Value *Main = Guard.CreateSub(TripV, Rem, "rdx.main");
  Value *End = Guard.CreateAdd(Start, Main, "rdx.end");
  Value *NGroups =
      Guard.CreateLShr(Main, ConstantInt::get(IVTy, P.ChainLog), "rdx.groups");
  Value *SkipMain = Guard.CreateICmpEQ(Main, ConstantInt::get(IVTy, 0));
  Guard.CreateCondBr(SkipMain, RemGuard, NarrowPH);
  PH->getTerminator()->eraseFromParent();

  for (unsigned OpI = 0; OpI < P.Inc->getNumOperands(); ++OpI) {
    if (auto *C = dyn_cast<ConstantInt>(P.Inc->getOperand(OpI))) {
      if (C->isOne())
        P.Inc->setOperand(OpI, ConstantInt::get(IVTy, P.Chains));
    }
  }
  dropWrap(P.Inc);
  dropWrap(P.RedOp);

  FastMathFlags FMF;
  if (isa<FPMathOperator>(P.RedOp))
    FMF = P.RedOp->getFastMathFlags();
  Value *Identity = getRecurrenceIdentity(P.Kind, P.RedPhi->getType(), FMF);
  const unsigned FoldOp = RecurrenceDescriptor::getOpcode(P.Kind);

  SmallVector<Value *, 8> Partials;
  Partials.push_back(P.RedOp);
  IRBuilder<> AtInc(P.Inc);
  for (unsigned K = 1; K < P.Chains; ++K) {
    PHINode *Part = PHINode::Create(P.RedPhi->getType(), 2,
                                    P.RedPhi->getName() + "." + Twine(K),
                                    H->begin());
    Value *IVK = AtInc.CreateAdd(P.IV, ConstantInt::get(IVTy, K),
                                 P.IV->getName() + "." + Twine(K));
    ValueToValueMapTy VM;
    VM[P.IV] = IVK;
    for (Instruction *I : P.Work) {
      Instruction *C = I->clone();
      C->setName(I->getName() + "." + Twine(K));
      C->insertBefore(P.Inc->getIterator());
      VM[I] = C;
      RemapInstruction(C, VM, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    }
    VM[P.RedPhi] = Part;
    Instruction *RedK = P.RedOp->clone();
    RedK->setName(P.RedOp->getName() + "." + Twine(K));
    RedK->insertBefore(P.Inc->getIterator());
    RemapInstruction(RedK, VM, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    dropWrap(RedK);
    Part->addIncoming(Identity, NarrowPH);
    Part->addIncoming(RedK, H);
    Partials.push_back(RedK);
  }

  PHINode *Groups =
      PHINode::Create(IVTy, 2, "rdx.grp", H->begin());
  Groups->addIncoming(ConstantInt::get(IVTy, 0), NarrowPH);
  Value *GroupsNext =
      AtInc.CreateAdd(Groups, ConstantInt::get(IVTy, 1), "rdx.grp.next");
  Groups->addIncoming(GroupsNext, H);
  Value *Stay = AtInc.CreateICmpULT(GroupsNext, NGroups, "rdx.stay");
  P.Cmp->replaceAllUsesWith(Stay);
  P.Cmp->eraseFromParent();
  BI->setSuccessor(0, H);
  BI->setSuccessor(1, MainExit);
  if (BI->getCondition() != Stay)
    BI->setCondition(Stay);

  // The tree fold uses the original reduction value. Snapshot the real
  // outside users first, so that fold is not rewritten to the final phi.
  SmallVector<Instruction *, 4> ExitUsers;
  for (User *U : P.RedOp->users()) {
    auto *UI = dyn_cast<Instruction>(U);
    if (UI && !L->contains(UI))
      ExitUsers.push_back(UI);
  }

  IRBuilder<> AtExit(MainExit);
  Value *Combined = Partials[0];
  for (unsigned K = 1; K < Partials.size(); ++K)
    Combined = combine(AtExit, Combined, Partials[K], FoldOp, P.RedOp);
  AtExit.CreateBr(RemGuard);

  IRBuilder<> AtGuard(RemGuard);
  PHINode *RemIVPhi = AtGuard.CreatePHI(IVTy, 2, "rem.iv");
  PHINode *RemSumPhi =
      AtGuard.CreatePHI(P.RedPhi->getType(), 2, "rem.sum");
  RemIVPhi->addIncoming(End, MainExit);
  RemIVPhi->addIncoming(Start, PH);
  RemSumPhi->addIncoming(Combined, MainExit);
  RemSumPhi->addIncoming(OrigInit, PH);
  Value *SkipRem = AtGuard.CreateICmpEQ(Rem, ConstantInt::get(IVTy, 0));
  AtGuard.CreateCondBr(SkipRem, Exit, RemPH);

  RemIV->setIncomingValueForBlock(RemPH, RemIVPhi);
  RemRed->setIncomingValueForBlock(RemPH, RemSumPhi);

  // The remainder needs its own exit. Sharing the original exit with the
  // skip edge would leave that block with a predecessor outside the loop.
  auto *RemBI = cast<BranchInst>(RemH->getTerminator());
  for (unsigned S = 0; S < RemBI->getNumSuccessors(); ++S)
    if (RemBI->getSuccessor(S) == Exit)
      RemBI->setSuccessor(S, RemExit);
  auto *RemLCSSA = PHINode::Create(P.RedPhi->getType(), 1, "rem.lcssa",
                                   RemExit->begin());
  RemLCSSA->addIncoming(RemExitVal, RemH);
  IRBuilder<>(RemExit).CreateBr(Exit);

  if (Result) {
    Result->removeIncomingValue(H, /*DeletePHIIfEmpty=*/false);
    Result->addIncoming(RemLCSSA, RemExit);
    Result->addIncoming(RemSumPhi, RemGuard);
  } else {
    auto *NewResult = PHINode::Create(P.RedPhi->getType(), 2, "split.rdx",
                                      Exit->begin());
    NewResult->addIncoming(RemLCSSA, RemExit);
    NewResult->addIncoming(RemSumPhi, RemGuard);
    for (Instruction *UI : ExitUsers)
      UI->replaceUsesOfWith(P.RedOp, NewResult);
  }

  L->setLoopAlreadyUnrolled();
  if (Loop *Rem = LI.getLoopFor(RemH))
    Rem->setLoopAlreadyUnrolled();

  DT.recalculate(*F);
  SE.forgetAllLoops();
  return true;
}

static bool runImpl(Function &F, LoopInfo &LI, DominatorTree &DT,
                    ScalarEvolution &SE) {
  if (!StarbugSplitAccumEnable)
    return false;
  const unsigned Lanes =
      std::max(1u, RISCVVLIW::StarbugVLIWConfig::fromCommandLine().NumLanes);

  SmallVector<Loop *, 8> Work;
  for (Loop *L : LI)
    for (Loop *Inner : depth_first(L))
      Work.push_back(Inner);

  bool Changed = false;
  for (Loop *L : Work) {
    if (!L->isInnermost())
      continue;
    SplitPlan P;
    if (!collectPlan(L, SE, Lanes, P))
      continue;
    Changed |= splitLoop(P, LI, DT, SE);
  }
  return Changed;
}

} // namespace

PreservedAnalyses llvm::StarbugSplitAccumPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (!TM->getSubtarget<RISCVSubtarget>(F).hasStarbugVLIW() || F.hasOptNone())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  if (!runImpl(F, LI, DT, SE))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
