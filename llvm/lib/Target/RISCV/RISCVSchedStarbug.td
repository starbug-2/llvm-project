//==- RISCVSchedStarbug.td - Starbug VLIW Scheduling ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In-order issue. StarbugLaneCap is IssueWidth and the StarbugIssue unit count
// together. The default is eight, the width of the compiler's machine.
// lane-sweep.sh rewrites this one number and rebuilds llc; -starbug-vliw-lanes
// alone leaves the scheduler at whatever this number was when llc was built.
//
// Every ProcResource has BufferSize 0, so a ReleaseAtCycles hold is pinned to
// the cycle the instruction issues. MicroOpBufferSize 0 is the in-order
// machine. A packet reads sources at the start of the cycle and writes
// integer ALU results at the end, so ALU latency stays 1. Loads, multiplies,
// and pipelined FP results are not forwarded from that same point, so their
// latency is 2. fdiv, fsqrt, and integer div/rem hold every issue slot for
// the divider's busy interval: hazard.sv stalls fetch, decode, and execute
// while DivBusyE or FDivBusyE is high.
//
// Those busy intervals are a hand walk of the FSMs, not a waveform. Integer
// div (RV32, 2 bits/cycle) is the start cycle plus 16 steps, 17 cycles, and
// only the lane-0 DivBusyE is wired to the hazard unit. Binary32 fdiv/fsqrt
// is CyclesE 7 plus the start cycle, 8 cycles (half 5, double 15, quad 30),
// from RK = 4 in the Starbug config. A dependent use can wait one more cycle
// after busy falls because the result is forwarded from writeback; the model
// uses the busy-high count.
//
// Two memory resources are permission to issue two memory operations in one
// cycle. TableGen cannot see address bit 2; the packetizer rejects a pair
// that resolves to the same bank. StarbugLane0 is the single slot shared by
// branches, CSRs, integer div/rem, and every write that is not listed below.
// That is the packetizer's lane-0 rule for branches and for opcodes it does
// not recognise. StarbugFPU is three units, the count the packetizer assumes.
// The Starbug configuration itself leaves F_SUPPORTED at 0 and does not
// instantiate those units.
//
defvar StarbugLaneCap = 8;
defvar StarbugIDivCycles = 17;
defvar StarbugFDivHCycles = 5;
defvar StarbugFDivSCycles = 8;
defvar StarbugFDivDCycles = 15;
defvar StarbugFDivQCycles = 30;

// CompleteModel checks every opcode in the RISC-V target. Features this core
// does not implement are skipped. A SchedWrite those opcodes already carry
// still receives the lane-0 default below, so enabling the feature later does
// not fall off the model. The kept set is the integer base plus the FP
// formats this file gives a real class.
defvar StarbugKeptFeatures = [HasStdExtM, HasStdExtZmmul,
                              HasStdExtA, HasStdExtZaamo, HasStdExtZalrsc,
                              HasAtomicLdSt,
                              HasStdExtZca, HasStdExtZcf, HasStdExtZcd,
                              HasStdExtZicsr, HasStdExtZifencei,
                              HasStdExtF, HasStdExtD, HasStdExtQ,
                              HasStdExtZfh, HasStdExtZfhmin, HasStdExtZfa,
                              HasStdExtZfhOrZvfh, HasStdExtFOrZfinx,
                              HasHalfFPLoadStoreMove,
                              HasStdExtZfinx, HasStdExtZdinx,
                              HasStdExtZhinx, HasStdExtZhinxmin,
                              HasStarbugVLIW];
defvar StarbugUnsupportedFeatures =
    !listremove(!filter(P, !instances<Predicate>(),
                        !eq(!substr(!cast<string>(P), 0, 3), "Has")),
                StarbugKeptFeatures);

def StarbugVLIWModel : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = StarbugLaneCap;
  let LoadLatency = 2;
  let HighLatency = StarbugIDivCycles;
  let MispredictPenalty = 1;
  let CompleteModel = 1;
  let UnsupportedFeatures = StarbugUnsupportedFeatures;
}

let SchedModel = StarbugVLIWModel in {
  // One record per issue slot, driven by StarbugLaneCap so lane-sweep.sh can
  // still rewrite that single number. Listing one ProcResource repeatedly does
  // not do this: TableGen adds those cycles together and holds a single unit
  // for LaneCap * N cycles, leaving the others free. Naming each unit holds
  // every slot for N cycles, which is the execute stall. StarbugIssue is the
  // group an ordinary op takes one slot of.
  let BufferSize = 0 in {
    foreach I = !range(StarbugLaneCap) in {
      def StarbugIssue#I : ProcResource<1>;
    }
    def StarbugLane0 : ProcResource<1>;
    def StarbugMem : ProcResource<2>;
    def StarbugFPU : ProcResource<3>;
  }
  defvar StarbugIssueUnits = !foreach(I, !range(StarbugLaneCap),
                                      !cast<ProcResource>("StarbugIssue" # I));
  let BufferSize = 0 in {
    def StarbugIssue : ProcResGroup<StarbugIssueUnits>;
  }

  defvar StarbugIDivRes = !listconcat(StarbugIssueUnits, [StarbugLane0]);
  defvar StarbugIDivRel = !listconcat(!listsplat(StarbugIDivCycles, StarbugLaneCap),
                                      [StarbugIDivCycles]);

  defvar StarbugALUWrites = [WriteIALU, WriteIALU32,
                             WriteShiftImm, WriteShiftImm32,
                             WriteShiftReg, WriteShiftReg32,
                             WriteNop];
  defvar StarbugMulWrites = [WriteIMul, WriteIMul32];
  defvar StarbugIDivWrites = [WriteIDiv, WriteIDiv32, WriteIRem, WriteIRem32];
  defvar StarbugLoadWrites = [WriteLDB, WriteLDH, WriteLDW, WriteLDD,
                              WriteFLD16, WriteFLD32, WriteFLD64, WriteFLD128,
                              WriteAtomicB, WriteAtomicH, WriteAtomicW, WriteAtomicD,
                              WriteAtomicLDW, WriteAtomicLDD];
  defvar StarbugStoreWrites = [WriteSTB, WriteSTH, WriteSTW, WriteSTD,
                               WriteFST16, WriteFST32, WriteFST64, WriteFST128,
                               WriteAtomicSTW, WriteAtomicSTD];
  // FP-to-integer moves forward in one cycle. Everything else that writes an
  // FP result takes the one-cycle FPUStallD bubble.
  defvar StarbugFMovToIWrites = [WriteFMovF16ToI16, WriteFMovF32ToI32,
                                 WriteFMovF64ToI64];
  defvar StarbugFPUWrites = [WriteFAdd16, WriteFAdd32, WriteFAdd64, WriteFAdd128,
                             WriteFMul16, WriteFMul32, WriteFMul64, WriteFMul128,
                             WriteFMA16, WriteFMA32, WriteFMA64, WriteFMA128,
                             WriteFCvtI32ToF16, WriteFCvtI32ToF32, WriteFCvtI32ToF64, WriteFCvtI32ToF128,
                             WriteFCvtI64ToF16, WriteFCvtI64ToF32, WriteFCvtI64ToF64, WriteFCvtI64ToF128,
                             WriteFCvtF16ToI32, WriteFCvtF16ToI64,
                             WriteFCvtF32ToI32, WriteFCvtF32ToI64,
                             WriteFCvtF64ToI32, WriteFCvtF64ToI64,
                             WriteFCvtF128ToI32, WriteFCvtF128ToI64,
                             WriteFCvtF32ToF64, WriteFCvtF64ToF32,
                             WriteFCvtF16ToF32, WriteFCvtF32ToF16,
                             WriteFCvtF16ToF64, WriteFCvtF64ToF16,
                             WriteFCvtF128ToF32, WriteFCvtF128ToF64,
                             WriteFCvtF32ToF128, WriteFCvtF64ToF128,
                             WriteFRoundF16, WriteFRoundF32, WriteFRoundF64, WriteFRoundF128,
                             WriteFClass16, WriteFClass32, WriteFClass64, WriteFClass128,
                             WriteFCmp16, WriteFCmp32, WriteFCmp64, WriteFCmp128,
                             WriteFSGNJ16, WriteFSGNJ32, WriteFSGNJ64, WriteFSGNJ128,
                             WriteFMinMax16, WriteFMinMax32, WriteFMinMax64, WriteFMinMax128,
                             WriteFMovI16ToF16, WriteFMovI32ToF32, WriteFMovI64ToF64,
                             WriteFLI16, WriteFLI32, WriteFLI64, WriteFLI128];
  defvar StarbugFDivHWrites = [WriteFDiv16, WriteFSqrt16];
  defvar StarbugFDivSWrites = [WriteFDiv32, WriteFSqrt32];
  defvar StarbugFDivDWrites = [WriteFDiv64, WriteFSqrt64];
  defvar StarbugFDivQWrites = [WriteFDiv128, WriteFSqrt128];

  defvar StarbugSpecialized =
      !listconcat(StarbugALUWrites, StarbugMulWrites, StarbugIDivWrites,
                  StarbugLoadWrites, StarbugStoreWrites,
                  StarbugFMovToIWrites, StarbugFPUWrites,
                  StarbugFDivHWrites, StarbugFDivSWrites,
                  StarbugFDivDWrites, StarbugFDivQWrites);

  foreach W = StarbugALUWrites in {
    def : WriteRes<W, [StarbugIssue]> { let Latency = 1; }
  }
  foreach W = StarbugMulWrites in {
    def : WriteRes<W, [StarbugIssue]> { let Latency = 2; }
  }
  foreach W = StarbugIDivWrites in {
    def : WriteRes<W, StarbugIDivRes> {
      let Latency = StarbugIDivCycles;
      let ReleaseAtCycles = StarbugIDivRel;
    }
  }
  foreach W = StarbugLoadWrites in {
    def : WriteRes<W, [StarbugMem, StarbugIssue]> { let Latency = 2; }
  }
  foreach W = StarbugStoreWrites in {
    def : WriteRes<W, [StarbugMem, StarbugIssue]> { let Latency = 1; }
  }
  foreach W = StarbugFMovToIWrites in {
    def : WriteRes<W, [StarbugFPU, StarbugIssue]> { let Latency = 1; }
  }
  foreach W = StarbugFPUWrites in {
    def : WriteRes<W, [StarbugFPU, StarbugIssue]> { let Latency = 2; }
  }
  foreach W = StarbugFDivHWrites in {
    def : WriteRes<W, StarbugIssueUnits> {
      let Latency = StarbugFDivHCycles;
      let ReleaseAtCycles = !listsplat(StarbugFDivHCycles, StarbugLaneCap);
    }
  }
  foreach W = StarbugFDivSWrites in {
    def : WriteRes<W, StarbugIssueUnits> {
      let Latency = StarbugFDivSCycles;
      let ReleaseAtCycles = !listsplat(StarbugFDivSCycles, StarbugLaneCap);
    }
  }
  foreach W = StarbugFDivDWrites in {
    def : WriteRes<W, StarbugIssueUnits> {
      let Latency = StarbugFDivDCycles;
      let ReleaseAtCycles = !listsplat(StarbugFDivDCycles, StarbugLaneCap);
    }
  }
  foreach W = StarbugFDivQWrites in {
    def : WriteRes<W, StarbugIssueUnits> {
      let Latency = StarbugFDivQCycles;
      let ReleaseAtCycles = !listsplat(StarbugFDivQCycles, StarbugLaneCap);
    }
  }

  // Branches, CSRs, and every opcode this file does not name. Lane 0 is one
  // slot, so they compete with each other.
  foreach W = !listremove(!instances<SchedWrite>(), StarbugSpecialized) in {
    def : WriteRes<W, [StarbugLane0, StarbugIssue]> {
      let Latency = 1;
    }
  }
  foreach R = !instances<SchedRead>() in {
    def : ReadAdvance<R, 0>;
  }

  // These have an empty SchedRW, so the write foreach never sees them.
  // WriteCSR is the lane-0 class. ECALL already uses WriteJmp.
  def : InstRW<[WriteCSR],
               (instrs FENCE, FENCE_TSO, FENCE_I, EBREAK, UNIMP,
                       SRET, MRET, MNRET, WFI, DRET,
                       SFENCE_W_INVAL, SFENCE_INVAL_IR, SFENCE_VMA,
                       SINVAL_VMA, HINVAL_VVMA, HINVAL_GVMA,
                       HFENCE_VVMA, HFENCE_GVMA, SCTRCLR,
                       HLV_B, HLV_BU, HLV_H, HLV_HU, HLVX_HU, HLV_W, HLVX_WU,
                       HLV_WU, HLV_D, HSV_B, HSV_H, HSV_W, HSV_D,
                       STARBUG_BUNDLE_HINT)>;

  // COPY is scheduled. It is a register move and may use any issue slot.
  def : InstRW<[WriteIALU], (instrs COPY)>;
  def : InstRW<[WriteIALU], (instrs ADJCALLSTACKDOWN, ADJCALLSTACKUP)>;
  def : InstRW<[WriteIALU],
               (instregex "^Pseudo(LI|LA$|LAImm|LA_TLS.*|LLA.*|LGA|AddTPRel|SEXT_.*|ZEXT_.*|BR.*|CALL.*|TAIL.*|RET|C_ADDI_NOP)$",
                          "^Select_")>;
  def : InstRW<[WriteLDW], (instregex "^PseudoL[BHWD]")>;
  def : InstRW<[WriteSTW], (instregex "^PseudoS[BHWD]")>;
  def : InstRW<[WriteFLD32], (instregex "^PseudoFL")>;
  def : InstRW<[WriteFST32], (instregex "^PseudoFS")>;
  def : InstRW<[WriteFCmp32], (instregex "^PseudoQuietF")>;
  def : InstRW<[WriteFRoundF32], (instregex "^PseudoFROUND")>;
  def : InstRW<[WriteAtomicW],
               (instregex "^Pseudo(Atomic|CmpXchg|Masked)")>;
  def : InstRW<[WriteFMovI32ToF32],
               (instregex "^BuildPairF64Pseudo$", "^SplitF64Pseudo$",
                          "^PseudoRV32Zdinx")>;
  def : InstRW<[WriteCSR],
               (instregex "^(Read|Write|Clear|Set|Swap)F(CSR|FLAGS|RM)",
                          "^WriteVXRMImm$", "^FMVH_X_Q$", "^FMVP_Q_X$")>;
  def : InstRW<[WriteCSR], (instrs HWASAN_CHECK_MEMACCESS_SHORTGRANULES)>;
  // SiFive matrix pseudos are declared with no predicate, so the feature
  // list above does not exclude them. They are not this machine.
  def : InstRW<[WriteCSR], (instregex "^PseudoSF_")>;
}
