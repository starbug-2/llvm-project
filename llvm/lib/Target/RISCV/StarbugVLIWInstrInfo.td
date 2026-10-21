// Packet length and the four predicate nibbles. The asm printer lowers this
// to SB2_HEADER. The length is not an operand of the header; it only selects
// how many of p0..p2 are set. A nibble of 0 is unconditional.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isPseudo = 1 in {
  def STARBUG_BUNDLE_HINT
      : Pseudo<(outs), (ins i32imm:$len, i32imm:$p0, i32imm:$p1, i32imm:$p2,
                            i32imm:$p3), []>;
}

// Predicate nibble: creg in bits [3:1], z in bit [0]. See RISCVSB2.
def SB2PredAsmOperand : AsmOperandClass {
  let Name = "SB2Pred";
  let ParserMethod = "parseSB2Pred";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isSB2Pred";
  let DiagnosticType = "InvalidSB2Pred";
  let DiagnosticString = "operand must be 'un', x9-x15, or !x9-!x15";
}

def sb2pred : RISCVOp {
  let ParserMatchClass = SB2PredAsmOperand;
  let PrintMethod = "printSB2Pred";
  let DecoderMethod = "decodeSB2Pred";
  let OperandType = "OPERAND_UIMM4";
}

// Printed in decimal so llvm-mc and llvm-objdump agree. objdump otherwise
// prints small immediates in hex.
def sb2parallel : RISCVUImmOp<3> {
  let PrintMethod = "printSB2Parallel";
}

// Starbug 2 execute-packet header. custom-0 major opcode 0001011.
//
//   31      28 27      24 23      20 19      16 15 14 13 12 11    7 6     0
//   creg z     creg z     creg z     creg z     p0 p1 p2 p3 reserved 0001011
//   op0        op1        op2        op3        parallel  5 bits
//
// Each predicate is one nibble, C6x order. creg 0 and z 0 is unconditional.
// creg 1..7 selects x9..x15. z 0 tests nonzero, z 1 tests zero.
// pi = 1 means operation i+1 issues in parallel with operation i.
// p3 is spare and reserved[11:7] is zero; both are fixed here.
// The following operations are not encoded in this word. Each declares its
// own width in inst[1:0].
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, hasNoSchedulingInfo = 1,
    DecoderNamespace = "Starbug", Predicates = [HasStarbugVLIW] in
def SB2_HEADER
    : RVInst<(outs),
             (ins sb2pred:$pred0, sb2pred:$pred1, sb2pred:$pred2,
                  sb2pred:$pred3, sb2parallel:$parallel),
             "sb2.header",
             "$pred0, $pred1, $pred2, $pred3, $parallel", [], InstFormatR> {
  bits<4> pred0;
  bits<4> pred1;
  bits<4> pred2;
  bits<4> pred3;
  bits<3> parallel;

  let Inst{31-28} = pred0;
  let Inst{27-24} = pred1;
  let Inst{23-20} = pred2;
  let Inst{19-16} = pred3;
  let Inst{15-13} = parallel;
  let Inst{12} = 0;
  let Inst{11-7} = 0;
  let Inst{6-0} = OPC_CUSTOM_0.Value;
}
