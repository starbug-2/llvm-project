//===-- StarbugNarrowIV.cpp - 32-bit latch for a 64-bit trip count --------===//
//
// On RV32 a 64-bit induction variable becomes a low half, a carry, and an
// extra branch. That splits the loop, so the steady state pays for a compare
// the trip count does not need when it fits in 32 bits. Version the loop:
// the wide copy stays correct for a huge trip count, and the copy that runs
// when the count fits uses one 32-bit compare.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "starbug-narrow-iv"

namespace {

class StarbugNarrowIV : public FunctionPass {
public:
  static char ID;
  StarbugNarrowIV() : FunctionPass(ID) {
    initializeStarbugNarrowIVPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
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
INITIALIZE_PASS_END(StarbugNarrowIV, DEBUG_TYPE,
                    "Narrow 64-bit loop IVs that fit in 32 bits", false, false)

// A one-block loop whose i64 IV is only used as a 32-bit index, plus a +1
// and an unsigned compare against a value defined outside the loop.
struct NarrowCandidate {
  Loop *L = nullptr;
  PHINode *IV = nullptr;
  TruncInst *Trunc = nullptr;
  BinaryOperator *Inc = nullptr;
  ICmpInst *Cmp = nullptr;
  Value *Start = nullptr;
  Value *Trip = nullptr;
  BasicBlock *Exit = nullptr;
};

static bool collectCandidate(Loop *L, NarrowCandidate &C) {
  if (!L->isInnermost() || L->getNumBlocks() != 1)
    return false;
  BasicBlock *H = L->getHeader();
  auto *BI = dyn_cast<BranchInst>(H->getTerminator());
  if (!BI || !BI->isConditional())
    return false;
  if (!((BI->getSuccessor(0) == H && BI->getSuccessor(1) == L->getExitBlock()) ||
        (BI->getSuccessor(1) == H && BI->getSuccessor(0) == L->getExitBlock())))
    return false;
  BasicBlock *Exit = L->getExitBlock();
  if (!Exit || !Exit->phis().empty())
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

  Value *Start = nullptr;
  BinaryOperator *Inc = nullptr;
  for (unsigned I = 0; I < 2; ++I) {
    Value *V = IV->getIncomingValue(I);
    BasicBlock *From = IV->getIncomingBlock(I);
    if (L->contains(From)) {
      Inc = dyn_cast<BinaryOperator>(V);
    } else {
      Start = V;
    }
  }
  if (!Start || !Inc || Inc->getOpcode() != Instruction::Add)
    return false;
  if (Inc->getParent() != H)
    return false;
  Value *Step = nullptr;
  if (Inc->getOperand(0) == IV)
    Step = Inc->getOperand(1);
  else if (Inc->getOperand(1) == IV)
    Step = Inc->getOperand(0);
  else
    return false;
  auto *StepC = dyn_cast<ConstantInt>(Step);
  if (!StepC || !StepC->isOne())
    return false;

  TruncInst *Trunc = nullptr;
  for (User *U : IV->users()) {
    if (U == Inc)
      continue;
    auto *T = dyn_cast<TruncInst>(U);
    if (!T || T->getDestTy() != Type::getInt32Ty(H->getContext()) ||
        T->getParent() != H)
      return false;
    if (Trunc)
      return false;
    Trunc = T;
  }
  if (!Trunc)
    return false;

  // The increment feeds the compare and comes back through the phi.
  ICmpInst *Cmp = nullptr;
  bool SawPhi = false;
  for (User *U : Inc->users()) {
    if (U == IV) {
      SawPhi = true;
      continue;
    }
    auto *I = dyn_cast<ICmpInst>(U);
    if (!I || Cmp)
      return false;
    Cmp = I;
  }
  if (!Cmp || !SawPhi || Cmp->getParent() != H || Cmp != BI->getCondition() ||
      Cmp->getPredicate() != ICmpInst::ICMP_ULT)
    return false;
  Value *Trip = Cmp->getOperand(0) == Inc ? Cmp->getOperand(1) : nullptr;
  if (!Trip)
    return false;
  if (auto *TripI = dyn_cast<Instruction>(Trip))
    if (L->contains(TripI->getParent()))
      return false;

  if (auto *StartC = dyn_cast<ConstantInt>(Start)) {
    if (!StartC->getValue().isIntN(32))
      return false;
  }

  C = {L, IV, Trunc, Inc, Cmp, Start, Trip, Exit};
  return true;
}

static bool narrowLoop(NarrowCandidate &C, LoopInfo &LI, DominatorTree &DT) {
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

  // The wide copy is the original 64-bit loop. The block we edit becomes
  // the 32-bit copy, taken when both the start and the trip count fit.
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> WideBlocks;
  Loop *Wide = cloneLoopWithPreheader(PH, PH, L, VMap, ".wide", &LI, &DT,
                                      WideBlocks);
  remapInstructionsInBlocks(WideBlocks, VMap);

  LLVMContext &Ctx = H->getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  IRBuilder<> Guard(PH->getTerminator());
  Value *Zero64 = ConstantInt::get(I64, 0);
  Value *TripWide = Guard.CreateICmpNE(Guard.CreateLShr(C.Trip, 32), Zero64);
  Value *Cond = TripWide;
  if (!isa<ConstantInt>(C.Start)) {
    Value *StartWide =
        Guard.CreateICmpNE(Guard.CreateLShr(C.Start, 32), Zero64);
    Cond = Guard.CreateOr(TripWide, StartWide);
  }
  Guard.CreateCondBr(Cond, Wide->getLoopPreheader(), H);
  PH->getTerminator()->eraseFromParent();

  Value *Start32 = nullptr;
  if (auto *StartC = dyn_cast<ConstantInt>(C.Start))
    Start32 = ConstantInt::get(I32, StartC->getZExtValue());
  else {
    IRBuilder<> AtStart(PH->getTerminator());
    Start32 = AtStart.CreateTrunc(C.Start, I32);
  }
  IRBuilder<> AtTrip(PH->getTerminator());
  Value *Trip32 = AtTrip.CreateTrunc(C.Trip, I32);

  PHINode *NewIV = PHINode::Create(I32, 2, C.IV->getName() + ".n", H->begin());
  IRBuilder<> AtInc(C.Inc);
  Value *NewInc =
      AtInc.CreateAdd(NewIV, ConstantInt::get(I32, 1), C.Inc->getName(),
                      /*HasNUW=*/true, /*HasNSW=*/false);
  NewIV->addIncoming(Start32, PH);
  NewIV->addIncoming(NewInc, H);

  C.Trunc->replaceAllUsesWith(NewIV);
  IRBuilder<> AtCmp(C.Cmp);
  Value *NewCmp = AtCmp.CreateICmpULT(NewInc, Trip32, C.Cmp->getName());
  C.Cmp->replaceAllUsesWith(NewCmp);

  if (DT.getNode(C.Exit) && DT.getNode(C.Exit)->getIDom()->getBlock() == H)
    DT.changeImmediateDominator(C.Exit, PH);

  // The old phi and the increment use each other, so drop the increment's
  // uses before deleting either one.
  C.Cmp->eraseFromParent();
  C.Inc->replaceAllUsesWith(PoisonValue::get(I64));
  C.Inc->eraseFromParent();
  C.Trunc->eraseFromParent();
  C.IV->eraseFromParent();
  return true;
}

bool StarbugNarrowIV::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  SmallVector<Loop *, 8> Work;
  for (Loop *L : LI)
    for (Loop *Inner : depth_first(L))
      Work.push_back(Inner);

  bool Changed = false;
  for (Loop *L : Work) {
    NarrowCandidate C;
    if (!collectCandidate(L, C))
      continue;
    Changed |= narrowLoop(C, LI, DT);
  }
  return Changed;
}

FunctionPass *llvm::createStarbugNarrowIVPass() { return new StarbugNarrowIV(); }
