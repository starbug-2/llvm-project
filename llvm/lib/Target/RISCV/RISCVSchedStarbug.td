//==- RISCVSchedStarbug.td - Starbug VLIW Scheduling ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eight-wide in-order issue. A packet reads sources at the start of the cycle
// and writes results at the end, so every modeled latency is 1.
//
// Two memory resources are permission to issue two memory operations in one
// cycle. TableGen cannot see address bit 2; the packetizer rejects a pair
// that resolves to the same bank. Integer arithmetic, shifts, logic,
// compares, moves, branches, and the compressed forms of those take an
// issue slot.
//
// A WriteRes for any one class makes tablegen require a resource for every
// RISC-V sched class, including vector. The foreach below marks every other
// write unsupported so those classes keep the default one-cycle latency and
// do not grow a hand-written vector model. It is not a resource model.

def StarbugVLIWModel : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = 8;
  let LoadLatency = 1;
  let MispredictPenalty = 1;
  let CompleteModel = 0;
}

let SchedModel = StarbugVLIWModel in {
  def StarbugIssue : ProcResource<8>;
  def StarbugMem : ProcResource<2>;

  def StarbugIssueWrite : SchedWriteRes<[StarbugIssue]> { let Latency = 1; }
  def StarbugMemWrite : SchedWriteRes<[StarbugMem]> { let Latency = 1; }

  foreach W = !instances<SchedWrite>() in {
    def : WriteRes<W, []> {
      let Latency = 1;
      let Unsupported = 1;
    }
  }
  foreach R = !instances<SchedRead>() in {
    def : ReadAdvance<R, 0>;
  }

  def : InstRW<[StarbugMemWrite],
               (instrs LB, LH, LW, LBU, LHU, SB, SH, SW,
                       C_LW, C_LWSP, C_SW, C_SWSP)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs MUL, MULH, MULHSU, MULHU, MULW,
                       DIV, DIVU, DIVW, DIVUW, REM, REMU, REMW, REMUW)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs ADD, ADDI, SUB, ADDW, ADDIW, SUBW,
                       C_ADD, C_ADDI, C_ADDI16SP, C_ADDI4SPN, C_SUB,
                       C_ADDW, C_SUBW)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs SLL, SLLI, SRL, SRLI, SRA, SRAI,
                       SLLW, SLLIW, SRLW, SRLIW, SRAW, SRAIW,
                       C_SLLI, C_SRLI, C_SRAI)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs XOR, XORI, OR, ORI, AND, ANDI,
                       C_XOR, C_OR, C_AND, C_ANDI)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs SLT, SLTI, SLTU, SLTIU, LUI, C_LUI, C_LI, C_MV)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs BEQ, BNE, BLT, BGE, BLTU, BGEU, C_BEQZ, C_BNEZ)>;
}
