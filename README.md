# oxide-sandbox

> An experimental crucible for the Flux→MIR→PTX compile path, with ternary conservation verification.

## Background Theory

The SuperInstance ecosystem aspires to a world where agents write GPU kernels in a high-level, verification-friendly intermediate language called Flux, and those kernels execute as PTX on NVIDIA GPUs. The gap between Flux's semantic world and GPU machine code is wide: type systems differ, memory models differ, and the correctness invariants that hold in Flux may be violated by aggressive GPU optimization.

`oxide-sandbox` is not a production compiler. It is a **scientific instrument** for exploring that gap. It asks: given a sequence of Flux bytecode operations, can we produce a synthetic MIR representation that is both optimizable and verifiable? Can we detect when ternary operations (the `{-1, 0, +1}` arithmetic at the heart of the SuperInstance worldview) preserve their conservation laws through compilation?

The theoretical commitment is to **semantic preservation under transformation**. Every stage of the sandbox pipeline must preserve the observable behavior of the source program, and the final PTX must be inspectable for conservation properties.

## How It Works

`oxide-sandbox` exposes an `ExperimentalCompiler` that runs a five-stage pipeline for every `CompileExperiment`:

| Stage | Purpose | Output |
|---|---|---|
| `parse_bytecode` | Accept a slice of `FluxOp` instructions. | Parsed operation vector. |
| `type_inference` | Walk operations and assign MIR types (`I32`, `F32`, `Ternary`, `GpuPtr`). | Type map keyed by register. |
| `mir_generation` | Lower Flux operations into `SyntheticMir` blocks containing `MirOp`s. | MIR blocks, register types, GPU metadata. |
| `ptx_generation` | Emit PTX-like text from MIR. | Simulated PTX lines, register count, barrier count. |
| `conservation_verify` | Check that ternary operations (`TADD`, `TMUL`) compose correctly in Z₃. | Conservation pass/fail verdict. |

The pipeline records timing and output size for each stage, producing an `ExperimentResult` with:

- Total duration and stages passed.
- Whether PTX was generated and its estimated size.
- Register and shared-memory usage.
- A `thread_coherence` score.
- A `conservation_verified` flag.

Flux operations include standard integer arithmetic (`ADD`, `SUB`, `MUL`), ternary arithmetic (`TADD`, `TMUL`), GPU indexing (`ThreadIdx`), synchronization (`SyncThreads`), memory access (`Load`, `Store`), and control flow (`Branch`, `Halt`).

## Experiments

The test suite encodes three canonical experiments:

### Simple Integer Compile

```rust
let ops = vec![
    FluxOp::MOVI { reg: 0, imm: 42 },
    FluxOp::MOVI { reg: 1, imm: 10 },
    FluxOp::ADD { rd: 2, rs1: 0, rs2: 1 },
    FluxOp::Halt,
];
```

Validates that basic integer code parses, types, lowers, emits PTX, and passes conservation trivially.

### Ternary Compile

```rust
let ops = vec![
    FluxOp::MOVI { reg: 0, imm: 1 },
    FluxOp::MOVI { reg: 1, imm: -1 },
    FluxOp::TADD { rd: 2, ra: 0, rb: 1 },
    FluxOp::TMUL { rd: 3, ra: 2, rb: 0 },
    FluxOp::Halt,
];
```

Validates that `{-1, 0, +1}` arithmetic is recognized as `MirType::Ternary` and that the conservation verifier confirms closure in Z₃.

### GPU Kernel Compile

A kernel using `ThreadIdx`, `TADD`, and `SyncThreads`. Validates that GPU-specific operations produce correct MIR and PTX barrier instructions.

### Large-Scale Experiment

A proposed experiment: randomly generate 10,000 Flux programs of varying ternary density and measure:

- Conservation verification rate.
- Average compile time per program.
- Register pressure as a function of ternary operation count.
- Correlation between thread coherence and final PTX size.

## Applications

- **Compiler research**: Test new lowering strategies before they enter the production `cuda-oxide` path.
- **Agent-generated code validation**: Verify that LLM-generated Flux kernels produce well-formed MIR before deployment.
- **Curriculum tool**: Teach GPU compilation stages with a minimal, instrumented pipeline.
- **Construct integration**: Validate constructs from `oxide-constructs` before they are deployed to live GPUs.
- **Regression detection**: Run the sandbox on every commit to catch semantic drift in the Flux language.

## Open Questions

1. **Conservation under memory operations**: The current verifier only checks ternary register use. Does loading a ternary value from global memory and storing it back preserve conservation across address spaces?
2. **Thread divergence and coherence**: How should `thread_coherence` be defined in the presence of divergent branches?
3. **Optimization soundness**: If we add MIR-level constant folding, does it preserve ternary conservation for all Z₃ expressions?
4. **PTX emission fidelity**: How closely must simulated PTX match real PTX for the sandbox to be a valid proxy?

## Cross-Links

- [SuperInstance agent-knowledge / FLUX-TO-PTX.md](https://github.com/SuperInstance/agent-knowledge/blob/main/FLUX-TO-PTX.md) — The theoretical bridge that `oxide-sandbox` experiments on.
- [SuperInstance agent-knowledge / TERNARY-NUMBERS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TERNARY-NUMBERS.md) — The `{-1, 0, +1}` arithmetic foundation.
- [SuperInstance agent-knowledge / CONSERVATION-LAWS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/CONSERVATION-LAWS.md) — Invariants the sandbox verifies.
- [SuperInstance agent-knowledge / GPU-AS-MOTOR-CORTEX.md](https://github.com/SuperInstance/agent-knowledge/blob/main/GPU-AS-MOTOR-CORTEX.md) — Why GPU compilation matters for agentic cognition.
- `cuda-oxide` — Production GPU runtime that consumes PTX from this pipeline.
- `oxide-constructs` — Source of Flux kernels fed into the sandbox.

## Quick Start

```rust
use oxide_sandbox::{ExperimentalCompiler, FluxOp, FluxSource};

let mut compiler = ExperimentalCompiler::new();
let ops = vec![
    FluxOp::MOVI { reg: 0, imm: 1 },
    FluxOp::MOVI { reg: 1, imm: -1 },
    FluxOp::TADD { rd: 2, ra: 0, rb: 1 },
    FluxOp::Halt,
];

let result = compiler.run_experiment(
    "ternary_test",
    &ops,
    FluxSource::AgentGenerated {
        model: "gpt-4".into(),
        intent: "compute ternary sum".into(),
    },
);

assert!(result.ptx_generated);
assert!(result.conservation_verified);
println!("Compiled in {} us across {} stages", result.total_duration_us, result.stages_passed);
```
