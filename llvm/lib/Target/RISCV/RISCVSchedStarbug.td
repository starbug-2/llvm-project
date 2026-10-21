//==- RISCVSchedStarbug.td - Starbug VLIW Scheduling ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Four-wide in-order issue. A packet reads sources at the start of the cycle
// and writes results at the end, so every modeled latency is 1.
//
// Integer load and store take the one memory resource. Integer multiply, add,
// and branch take an issue slot. InstRW is limited to those opcodes.
//
// A WriteRes for any one class makes tablegen require a resource for every
// RISC-V sched class, including vector. The foreach below marks every other
// write unsupported so those classes keep the default one-cycle latency and
// do not grow a hand-written vector model. It is not a resource model.

def StarbugVLIWModel : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = 4;
  let LoadLatency = 1;
  let MispredictPenalty = 1;
  let CompleteModel = 0;
}

let SchedModel = StarbugVLIWModel in {
  def StarbugIssue : ProcResource<4>;
  def StarbugMem : ProcResource<1>;

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
               (instrs LB, LH, LW, LBU, LHU, SB, SH, SW)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs MUL, MULH, MULHSU, MULHU, ADD, ADDI)>;
  def : InstRW<[StarbugIssueWrite],
               (instrs BEQ, BNE, BLT, BGE, BLTU, BGEU)>;
}
