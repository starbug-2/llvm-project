// Packet length, the shared predicate register, and one sense bit per lane.
// The asm printer lowers this to SB2_HEADER. creg 0 is unconditional. The
// sense field's bit 7 is lane 0 (1 tests zero). The lane mask is a dense
// prefix of $len and is not stored on the pseudo.
let hasSideEffects = 1, mayLoad = 0, mayStore = 0, isPseudo = 1 in {
  def STARBUG_BUNDLE_HINT
      : Pseudo<(outs), (ins i32imm:$len, i32imm:$creg, i32imm:$sense), []>;
}

// Shared predicate register. 0 prints as 'un'. 1..7 name x9..x15.
def SB2CRegAsmOperand : AsmOperandClass {
  let Name = "SB2CReg";
  let ParserMethod = "parseSB2CReg";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isSB2CReg";
  let DiagnosticType = "InvalidSB2CReg";
  let DiagnosticString = "predicate register must be 'un' or x9-x15";
}

def sb2creg : RISCVOp {
  let ParserMatchClass = SB2CRegAsmOperand;
  let PrintMethod = "printSB2CReg";
  let DecoderMethod = "decodeSB2CReg";
  let OperandType = "OPERAND_UIMM3";
}

// Printed in decimal so llvm-mc and llvm-objdump agree. objdump otherwise
// prints small immediates in hex. Bit 7 is lane 0.
def sb2field8 : RISCVUImmOp<8> {
  let PrintMethod = "printSB2Imm";
}

// Lane mask. Bit 7 is lane 0. Zero names no lane, so it is not a header.
def SB2MaskAsmOperand : AsmOperandClass {
  let Name = "SB2Mask";
  let RenderMethod = "addImmOperands";
  let PredicateMethod = "isSB2Mask";
  let DiagnosticType = "InvalidSB2Mask";
  let DiagnosticString = "lane mask must be an integer in the range [1, 255]";
}

def sb2mask : RISCVOp {
  let ParserMatchClass = SB2MaskAsmOperand;
  let PrintMethod = "printSB2Imm";
  let DecoderMethod = "decodeSB2Mask";
  let OperandType = "OPERAND_UIMM8";
  let EncoderMethod = "getImmOpValue";
}

// Starbug 2 execute-packet header. custom-0 major opcode 0001011.
//
//   31   29 28          21 20    16 15       8 7 6     0
//   creg    sense           00000  lane mask  0 0001011
//
// creg 0 is unconditional. creg 1..7 selects x9..x15 and gates every occupied
// lane. Sense bit 28 (bit 7 of the field) is lane 0: 0 tests nonzero, 1 tests
// zero. Mask bit 15 (bit 7 of the field) set means lane 0 is occupied. The
// following operations are packed in lane order. Each declares its own width
// in inst[1:0]. Bits [20:16] and bit 7 are zero.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, hasNoSchedulingInfo = 1,
    DecoderNamespace = "Starbug", Predicates = [HasStarbugVLIW] in
def SB2_HEADER
    : RVInst<(outs), (ins sb2creg:$creg, sb2field8:$sense, sb2mask:$mask),
             "sb2.header", "$creg, $sense, $mask", [], InstFormatR> {
  bits<3> creg;
  bits<8> sense;
  bits<8> mask;

  let Inst{31-29} = creg;
  let Inst{28-21} = sense;
  let Inst{20-16} = 0;
  let Inst{15-8} = mask;
  let Inst{7} = 0;
  let Inst{6-0} = OPC_CUSTOM_0.Value;
}
