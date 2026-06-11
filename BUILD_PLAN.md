# BUILD PLAN — Foundation Model Distributed Training with Ray

> **Read this whole document before touching anything.** It is the build brief
> for assembling a standalone, self-contained **Ray Summit 2026** half-day
> training course from material that already exists on this workspace. Most of
> the work is *curation, re-sequencing, restyling, T4-retuning, and narrative
> tying* — not writing new ML logic. The hard ML/infra code already exists in
> the source tutorials (FSDP, DeepSpeed, fault-tolerance, observability) and the
> theory exists in the MIT lecture. Our job is to weave them into one coherent,
> pre-run, take-home course that (1) sells Ray as the best orchestrator and
> (2) positions Anyscale as the leading expert org on distributed FM training.

---

## 1. Mission

Assemble an **instructor-led, take-home notebook course** titled **"Foundation
Model Distributed Training with Ray"** (instructor: **Ian Jordan**) into
`/home/ray/default/Ray_Summit_2026_Foundation_Model_Training/`.

Audience: **several hundred** attendees in a **3-hour morning session** with one
break. **Attendees follow along live on their own clusters** (Anyscale/AWS
handle provisioning — out of scope for us). The instructor chooses depth live;
the notebooks must **stand alone afterward** as the durable artifact.

The course teaches how to train foundation models across many GPUs on
**Ray + Anyscale**, through one narrative spine:

> **You want to train a foundation model.** One GPU can't hold it →
> **Ray Train orchestrates many GPUs with ~no code change** → the
> model + optimizer + activations don't fit → **shard them (FSDP / DeepSpeed
> ZeRO)** → even sharded, one GPU's compute can't hold the tensors →
> **split them (Tensor + Data parallelism in 2D; Pipeline / Expert as the next
> axes)** → at scale things fail and you're blind → **Ray + Anyscale give fault
> tolerance, elastic recovery, and deep observability** → real runs combine all
> of it → **here's how LFM2 was actually trained.**

The pedagogical through-line: **one Ray surface** (`TorchTrainer` +
`prepare_model`/`get_dataset_shard` + `ray.train.report` + `ScalingConfig` +
`RunConfig`/`FailureConfig`) carries unchanged from a 4-GPU DDP job, to sharded
data parallelism, to 2D tensor+data parallelism, to fault-tolerant production
runs — **change the config, not the code.** The framework-agnostic theory (the
MIT lecture) is the credibility layer: we understand the entire landscape.

It maps 1:1 to the official course outline bullets:
- Data, tensor, pipeline, and expert parallelism
- FSDP and DeepSpeed training architectures
- Multi-node GPU scheduling and orchestration
- Checkpointing and fault recovery
- Distributed data sharding and ingestion with Ray Data
- Training throughput and utilization optimization

---

## 2. Hard constraints (do not violate)

1. **Environment: Anyscale workspace, 8-CPU head + one `g4dn.12xlarge` worker =
   4× NVIDIA T4 (16 GB each), 48 vCPU, 192 GiB.** Image
   `anyscale/ray:2.55.1-py311-cu128` (Ray 2.55.1, Python 3.11, CUDA 12.8). All
   notebooks must run and produce committed outputs on **4× T4**. **4 GPUs is
   the constant** — the day-of cluster may become **2 nodes × 2 GPUs**, so do
   **not** hardcode single-node assumptions; let Ray place workers.
2. **T4 = fp16, NOT bf16.** T4 is Turing (compute capability 7.5) with no bf16.
   Every mixed-precision path must select fp16 via
   `torch.cuda.is_bf16_supported()` (the source tutorials already do this).
   Teach this as a real hardware lesson, not a workaround to hide.
3. **g4dn T4s use PCIe, NOT NVLink.** Do not claim NVLink on this hardware.
   Frame TP honestly: "TP wants NVLink in production; here it rides PCIe, which
   is exactly why we keep TP-degree small (2)." Honesty == expertise == goal #2.
4. **Notebooks ship CLEAN (no committed outputs).** Decision changed 2026-06-11
   (Ian's call). Every notebook must still **execute cleanly top-to-bottom on
   this cluster** — verify by running it during the build — but the committed
   deliverable has outputs stripped (`outputs: []`, `execution_count: null`), so
   attendees run each cell fresh. Use **smoke-scale configs** (small step counts)
   so the verification run finishes in minutes. Workflow: build executes to
   validate, then strip outputs before shipping.
5. **Standalone & self-contained.** Notebooks may cross-reference *each other
   only* (`prerequisite_00` and `00`–`05`). No references to external courses,
   the robotics course, or the source tutorial directories.
6. **Notebook-only deliverable.** Markdown cells are the slides. No separate
   slide deck, no instructor run-of-show file.
7. **Unified house style** across all notebooks (see §6), mirroring
   `/home/ray/default/ray_summit_robotics_2026`.
8. **No Weights & Biases anywhere.** Metrics flow through Ray Train.
9. **Ray Train V2.** Set `RAY_TRAIN_V2_ENABLED=1` (env var + worker runtime_env)
   everywhere, matching the source tutorials.
10. **PP & EP are conceptual for now.** Pipeline and expert parallelism cannot
    train meaningfully on 4× T4; teach them with diagrams + the MIT mental
    models (03 + 05). Leave the door open to small runnable toys later, but do
    not block the build on them.

---

## 3. Source material inventory (lift / curate from here)

All paths absolute on this workspace. **Lift the working code; restyle the prose.**

| Source | Use for |
|---|---|
| `MIT_lecture/6S191_MIT_DeepLearning_L9.pdf` + `lecture_info.txt` | Theory for **00** (why/what scale, memory wall, 5 axes), **02** (memory math, ZeRO/FSDP), **03** (TP/PP/EP mental models), **05** (5D synthesis, LFM2 case study). Slides are image-based (render via pymupdf). We own these — reuse figures (see §3.1). Attribute the lecturer (Mathias Lechner, Liquid AI / MIT). |
| `ray_train_and_data_tutorials/prerequisite_00_ray_data.ipynb` | → `prerequisite_00_ray_data.ipynb` (restyle + T4-tune + pre-run). |
| `ray_train_and_data_tutorials/prerequisite_01_ray_train.ipynb` | **Absorbed into `01_ray_foundations.ipynb`** (single-GPU→DDP migration, TorchTrainer, Ray Data ingest, observability/fault-tolerance teasers). Not shipped as a separate prereq. |
| `fsdp_tutorial_1/README.md` (+ `images/`) | → **02** (FSDP2 data-parallel sharding, CPU offload, reshard, mixed precision, DCP checkpointing, PyTorch memory profiler). |
| `deepspeed_tutorial_1/README.md` (+ `train (1).py`) | → **02** (DeepSpeed ZeRO stages 1/2/3, `deepspeed.initialize`, save/load checkpoint). |
| `fsdp_tutorial_2/README.md` (+ `images/tensor_parallel.png`) | → **03** (DTensor `parallelize_module` Colwise/Rowwise + FSDP2 over the DP mesh dim, 2D `init_device_mesh`, TP-aware data loading). |
| `deepspeed_tutorial_2/README.md` (+ `images/tp_partition.png`) | → **03** (DeepSpeed AutoTP config-driven TP + MPU + ZeRO 1/2). |
| `ray_train_observability_and_debugging/notebook.ipynb` | → **04** (Ray Train dashboard, torch profiler views, throughput monitoring, Anyscale custom Train dashboard + dynolog on-demand GPU profiling). |
| `fault_tolerance_with_ray_train/01_Basic_Fault_Tolerance.ipynb` + `kill_train_worker (1).py` | → **04** (FailureConfig auto-retry, checkpoint-aware loop, live SIGKILL demo, manual restoration). Copy the kill util into `common/kill_train_worker.py` (drop the ` (1)` suffix). |
| `ray+megatron.txt` → <https://docs.anyscale.com/tutorials/train-with-megatron> | **Reference only** (no runnable Megatron here): Anyscale **Megatron-Bridge + Ray Train** SFT of Qwen2.5-1.5B on 8 GPUs (8×H100 or 2×4×L4), Ray Train as the orchestration layer. Cite in **03** (native TP/PP framework) and **05** (framework comparison). |

### 3.1 Image reuse policy
We own all course assets. Reuse:
- The FSDP/DeepSpeed tutorial PNGs (memory profiles, `tensor_parallel.png`,
  `tp_partition.png`) — copy into `images/`.
- Ray/Anyscale public S3 + docs diagrams already referenced by the source
  tutorials (checkpoint lifecycle, Ray Train V2 architecture, worker-failure
  recovery sequence, dashboards) — reference by URL as the tutorials do, or
  download into `images/` for offline robustness.
- MIT lecture figures may be re-rendered/redrawn as original ASCII or images for
  00/05; attribute the source. Prefer our own ASCII diagrams in the house style.

---

## 4. Target repo layout

```
Ray_Summit_2026_Foundation_Model_Training/
├── BUILD_PLAN.md                                  # this file
├── README.md                                      # course index + arc + prereqs + cluster image (NEW)
├── prerequisite_00_ray_data.ipynb                 # optional pre-reading: Ray Data essentials
├── 00_scaling_imperative.ipynb                    # why/what scale, 5 axes, memory wall (theory)
├── 01_ray_foundations.ipynb                       # Ray Train + Data: single-GPU→distributed DDP
├── 02_memory_and_sharding.ipynb                   # activation ckpt, fp16, offload, FSDP + DeepSpeed ZeRO
├── 03_2d_parallelism.ipynb                        # TP+DP via DTensor & AutoTP (2×2); PP/EP conceptual
├── 04_fault_tolerance_and_observability.ipynb     # FailureConfig + live kill, elastic; dashboard/profiling
├── 05_putting_it_together_lfm2.ipynb              # 5D synthesis + LFM2 case study
├── common/
│   ├── __init__.py
│   ├── kill_train_worker.py                       # SIGKILL a RayTrainWorker (from fault-tolerance tut)
│   └── utils.py                                   # shared model/data/checkpoint/precision helpers
├── images/                                        # reused tutorial PNGs + our diagrams
└── requirements.txt                               # canonical pinned deps (installed cluster-wide)
```

> **Dependency strategy (verified on the cluster, 2026-06-11).** torch is NOT in
> the base image — install the full stack with **plain `pip install`** (no
> `--index-url`; the default PyPI torch 2.9.x wheel is a cu128 build). On an
> Anyscale workspace, plain installs **register on all cluster nodes**
> automatically — a custom `--index-url` is NOT propagated (Anyscale warns). So
> notebooks do **not** pip-install in `runtime_env`; they use only
> `runtime_env={"working_dir": ".", "env_vars": {...}}` (matching the robotics
> course). `requirements.txt` is the canonical spec the day-of image/workspace
> installs. **No uv.** No per-job lockfile.

---

## 5. Datasets & models referenced

| Asset | Where | Used by | Notes |
|---|---|---|---|
| `gpt2` (124M) | HF | 01, 02 | Smallest LLM; DDP foundations + ZeRO/FSDP sharding hero. |
| `Qwen/Qwen2.5-0.5B` | HF | 03 | 2D-parallelism hero. `num_key_value_heads=2` ⇒ **`tp=2, dp=2` is the natural max on 4 GPUs** (`tp_size` must divide num_kv_heads). |
| ResNet18 / MNIST | torchvision | 01 (optional), 04 | Live-kill fault-tolerance demo (fast, robust). |
| ViT / FashionMNIST or CIFAR-10 | torchvision | 02 (memory-profile aside), 04 (observability) | Keep where the memory-profiler / profiler visuals add pedagogy (Mixed-hero decision). |
| `ag_news`, `wikitext-2-raw-v1` | HF datasets | 02, 03 | Small text corpora; use `train[:N%]` slices. |

Storage: `/mnt/cluster_storage/...` (shared, all nodes — checkpoints/results) and
`/mnt/local_storage/...` (per-node staging). Datasets are small enough to
download per run; no special staging needed (unlike the robotics course).

---

## 6. House style (apply to ALL notebooks)

Match `/home/ray/default/ray_summit_robotics_2026` notebooks exactly:

- **Top intro block (markdown):** `# Title`, then `## TLDR`, `## Introduction`,
  `## Key concepts used in this notebook`, `## What you will learn`,
  `## Why Ray on Anyscale for <X>?` (a **Challenge | Without Ray | With Ray**
  comparison table), an `## Architecture` ASCII diagram, and a
  `## How this scales on Anyscale` table (this-tutorial vs production).
- **Each code cell preceded by a `## Cell N — <title>` markdown header** with
  three sub-sections written as bold labels ending in a period. **What you do.**
  / **What to check.** / **Why it matters.** Align assignments in code; comment
  every non-obvious env var.
- **PROSE PUNCTUATION RULE.** In all natural-language prose, **no em-dashes (—),
  no semicolons (;), no colons (:)**. They read as AI-written. Rewrite with
  periods, commas, or parentheses. For a list lead-in, end the sentence with a
  period, then start the list (do not use a colon). Replace `Note:` with `Note
  that ...`. **Exception — section titles/headers may use em-dashes** (Ian's
  call), e.g. `## Cell 1 — Configuration`; still avoid colons in titles. The
  rule applies to markdown prose ONLY. Code, tables, URLs, and markdown syntax
  keep their colons. Lift-from-source prose must be rewritten to comply.
  Build executor uses `nbclient` (nbconvert is broken on this image).
- **`## Conclusion`** cell recapping the Ray primitives used and the scaling
  levers, plus a one-line forward reference to the next notebook.
- Rich **ASCII diagrams** and **LaTeX** blocks where they clarify (memory
  breakdown math, TP col/row, device-mesh grids, the 5D combination).
- **Honest scope framing**: smoke-scale runs validating *mechanics and
  infrastructure*; fp16/PCIe/4-GPU caveats stated plainly.
- Thread the two goals continuously: every notebook's `## Why Ray on Anyscale`
  table sells the orchestrator; theory depth sells the expertise.

---

## 7. Per-notebook build instructions

### prerequisite_00 — `prerequisite_00_ray_data.ipynb`
- **Purpose:** optional pre-reading — Ray Data essentials so a mixed-experience
  room shares a floor.
- **Source:** `ray_train_and_data_tutorials/prerequisite_00_ray_data.ipynb`.
- **Teach:** streaming execution, blocks/object store, lazy execution, `map` vs
  `map_batches`, stateful `ActorPoolStrategy` batch inference, materialize/
  persist, observability, shuffle/perf/fault-tolerance (advanced, lighter).
- **T4 config:** mostly CPU; the MNIST batch-inference example runs fine. Trim
  to the core pipeline; keep advanced sections as read-only appendix.
- **Outline bullet:** distributed data sharding & ingestion (foundation).

### 00 — `00_scaling_imperative.ipynb` (NEW, mostly markdown)
- **Purpose:** establish *why* we scale and *what* the levers are, before any
  Ray code. The expertise hook.
- **Source:** MIT lecture (slides 3–13, 18–22): GPU vs CPU FLOPs, scaling laws
  (Kaplan/Chinchilla), the memory breakdown (params/grads/optimizer/
  activations → ~28 GB for 1B, ~2 TB for 70B), activation checkpointing,
  offloading + bandwidth hierarchy, the 5 parallelism axes preview.
- **Teach:** the scaling imperative; the memory wall; preview DP/TP/PP/SP/EP and
  sharding (ZeRO/FSDP); bandwidth hierarchy (Cache > HBM > NVLink > IB > PCIe)
  and why it dictates strategy. End on: "Ray is how you orchestrate all of this."
- **Build:** markdown + ASCII diagrams + one tiny live cell:
  `ray.init(address="auto")` + `ray.cluster_resources()` to confirm 4× T4.
- **Outline bullet:** motivation for all four parallelism types.

### 01 — `01_ray_foundations.ipynb`
- **Purpose:** the orchestrator story — take a single-GPU PyTorch loop to a
  distributed 4-GPU DDP job with a handful of Ray calls. **Absorbs the Ray Train
  prereq.**
- **Source:** `ray_train_and_data_tutorials/prerequisite_01_ray_train.ipynb`
  (single-GPU baseline → `prepare_model` → Ray Data ingest via
  `get_dataset_shard`/`iter_torch_batches` → `ray.train.report` →
  `TorchTrainer`/`ScalingConfig`/`RunConfig` → inspect results → observability +
  fault-tolerance teasers).
- **Hero:** a small LLM (gpt2) fine-tune to keep the "foundation model" framing
  (fallback: keep ResNet18/MNIST if gpt2+RayData ingest is fiddly at smoke
  scale — Mixed decision allows it; prefer the LLM).
- **Teach:** controller/worker model; `prepare_model` (DDP); Ray Data sharding;
  `report` as a global barrier + rank-0 checkpoint; `ScalingConfig` (4→400 =
  one number); topology-agnostic placement (1×4 vs 2×2 — Ray's job, not yours).
- **T4 config:** `num_workers=4, use_gpu=True`, fp16 where applicable, small
  step count. Uses all 4 GPUs.
- **Outline bullets:** multi-node GPU scheduling & orchestration; data sharding.

### 02 — `02_memory_and_sharding.ipynb`
- **Purpose:** the model + optimizer + activations don't fit — shard them.
- **Source:** `fsdp_tutorial_1` (FSDP2 data-parallel: `fully_shard` per block,
  `CPUOffloadPolicy`, `reshard_after_forward`, `MixedPrecisionPolicy`, DCP
  `AppState`/`get_state_dict`, memory profiler) + `deepspeed_tutorial_1`
  (`deepspeed.initialize`, ZeRO stages 1/2/3, `save/load_checkpoint`,
  `get_precision_config` fp16 fallback).
- **Hero:** small LLM (gpt2) for both FSDP and DeepSpeed paths; **keep the ViT
  memory-profiler aside** from FSDP-1 where its GPU-memory-timeline visuals add
  teaching value (Mixed decision).
- **Teach:** the §00 memory math made real; activation checkpointing; fp16 on
  T4; CPU offload trade-off (PCIe!); FSDP sharding granularity + the three
  strategies; DeepSpeed ZeRO 1/2/3 with the per-stage memory/comm table; DCP vs
  DeepSpeed partitioned checkpoints; live memory profiling → HTML/`Files` tab.
- **T4 config:** `num_workers=4, use_gpu=True`. fp16. Small model + short run.
  Show ZeRO-3 / FULL_SHARD fitting a model that wouldn't fit replicated.
- **Outline bullet:** FSDP and DeepSpeed training architectures.

### 03 — `03_2d_parallelism.ipynb`
- **Purpose:** even sharded, split the tensors — TP + DP in 2D; PP/EP conceptual.
- **Source:** `fsdp_tutorial_2` (DTensor `parallelize_module` Colwise/Rowwise +
  FSDP2 over DP mesh, 2D `init_device_mesh`, TP-aware `DistributedSampler` by
  `dp_rank`, `foreach=False`, DCP + metadata.json) + `deepspeed_tutorial_2`
  (AutoTP `"tensor_parallel":{"autotp_size":N}`, manual TP/DP process groups +
  `ModelParallelUnit`, ZeRO 1/2 only). PP/EP from the MIT lecture (slides
  14, 17). Megatron-Bridge reference (Anyscale tutorial) as the native TP/PP
  framework.
- **Hero:** `Qwen/Qwen2.5-0.5B` (num_kv_heads=2 ⇒ `tp=2, dp=2` = 4 workers).
- **Teach:** TP column/row partitioning + the single all-reduce/layer; 2D device
  mesh + rank math (`tp_rank=wr%tp`, `dp_rank=wr//tp`); **TP-aware data loading**
  (shard by `dp_rank`, global batch = `bs × dp_size`) — the classic gotcha;
  DTensor+FSDP2 path vs DeepSpeed AutoTP path (and AutoTP's no-ZeRO-3 limit);
  when TP vs FSDP/ZeRO (comm-volume formulas, NVLink-vs-PCIe reality here);
  PP (bubble, micro-batching, DualPipe) and EP (router top-k, all-to-all) as
  diagrams; Megatron-Bridge as the production TP/PP option Anyscale supports.
- **T4 config:** `num_workers=4`, `tp_size=2`, `dp_size=2`, fp16, `seq_length`
  ≤512, `debug_steps≈20`. Both DTensor and AutoTP runs.
- **Outline bullet:** data, tensor, pipeline, and expert parallelism.

### 04 — `04_fault_tolerance_and_observability.ipynb`
- **Purpose:** at scale things fail and you fly blind — Ray + Anyscale fix both.
- **Source:** `fault_tolerance_with_ray_train/01_Basic_Fault_Tolerance.ipynb`
  (+ `common/kill_train_worker.py`) and
  `ray_train_observability_and_debugging/notebook.ipynb`.
- **Hero:** ResNet18/MNIST for the live-kill demo (fast, robust); ViT/CIFAR for
  the profiler/throughput section (its visuals are the tutorial's strength).
- **Teach:** checkpoint-aware loop (`get_checkpoint` + save model/optim/epoch);
  `FailureConfig(max_failures)` auto-retry → **live SIGKILL demo** via
  `kill_in()` → recovery from last checkpoint; manual restoration (same
  name+storage); elastic training (`num_workers=(min,max)`, spot, ~60% savings)
  and mid-epoch resumption (conceptual + snippet); then observability: Ray Train
  dashboard, throughput monitoring (linear scaling), torch profiler views
  (operator/trace/kernel/memory), and the **Anyscale-specific** persisted Train
  dashboard + **dynolog on-demand GPU profiling** (`KINETO_USE_DAEMON=1`).
- **T4 config:** `num_workers=2` for the kill demo (lose one, recover); 2 and 4
  workers for the throughput-scaling comparison. fp16. Short runs.
- **Outline bullets:** checkpointing & fault recovery; throughput/utilization
  optimization.

### 05 — `05_putting_it_together_lfm2.ipynb` (synthesis, mostly markdown)
- **Purpose:** combine everything; show a real production stack.
- **Source:** MIT lecture (slides 18–22): framework comparison
  (DeepSpeed/FSDP/Megatron), 5D parallelism, the 2048-GPU example
  (`DP8 × TP8 × PP4 × EP8`), LFM2 / LFM2-8B-A1B MoE case study, "match strategy
  to bandwidth hierarchy."
- **Teach:** how the memory layer (checkpointing, mixed precision, FSDP/ZeRO,
  offload) × the throughput layer (DP/TP/PP/SP/EP) compose; decision guidance
  (which axis when, tied to the bandwidth hierarchy); where Ray + Anyscale
  orchestrate the full stack (incl. Megatron-Bridge); the LFM2 anchor.
- **Build:** markdown + ASCII (the 5D grid) + a decision table. No heavy code;
  optionally a recap cell mapping each notebook → Ray primitive → outline bullet.
- **Outline bullet:** synthesis of all six.

---

## 8. Cross-reference / self-containment rules

- The course is standalone. **Strip every reference** to the source tutorial
  directories, the robotics course, "earlier in the series", external course
  names, or specific old filenames. Reference only `prerequisite_00` and
  `00`–`05` in this directory.
- Add *helpful* forward/back references: 01 → "the same `TorchTrainer` you'll
  reuse in 02/03/04"; 02 → "the memory math from 00, made real"; 03 → "FSDP from
  02 now sharded along a second axis"; 04 → "every run since 01 already called
  `report` — now we make it fault-tolerant"; 05 → ties all back.
- `00` and `README.md` establish the arc so references resolve.

---

## 9. Known gotchas & required fixes

1. **fp16 only (T4) — and the bf16-emulation trap.** Verified on this cluster:
   `torch.cuda.is_bf16_supported()` returns **True** on a T4 because recent
   PyTorch counts *software emulation*; `is_bf16_supported(including_emulation=
   False)` correctly returns **False** (T4 is sm_75, no native bf16). The naive
   helper would pick slow emulated bf16. **Use the native check**
   (`common.utils.has_native_bf16()` / `mixed_precision_dtype()`), which selects
   fp16 on T4. This is a teaching beat in 00/02. Apply fp16 everywhere (FSDP
   `MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16)`;
   DeepSpeed `{"fp16": {"enabled": True}}`; `torch.autocast` dtype fp16). Watch
   fp16 instability: rely on DeepSpeed's loss scaler and FSDP's reduce-in-fp16;
   keep LRs small (source tutorials use 1e-5/1e-6). If a pure-fp16 LLM run NaNs
   at smoke scale, lower LR / shorten run — do not switch hardware claims.
2. **PCIe, not NVLink.** g4dn T4s have no NVLink/P2P. Likely need
   `NCCL_P2P_DISABLE=1` (and possibly `NCCL_IB_DISABLE=1`,
   `NCCL_SHM_DISABLE=1`) for clean NCCL init on this containerized cluster —
   verify during build (the robotics course sets these). Never imply NVLink
   bandwidth on this hardware.
3. **TP degree.** `tp_size` must divide the model's `num_key_value_heads`.
   Qwen2.5-0.5B has 2 ⇒ max `tp_size=2` on this model. State this explicitly in
   03 so attendees who bump `tp_size` understand the error.
4. **AutoTP ≠ ZeRO-3.** DeepSpeed AutoTP supports ZeRO stage 1/2 only; for full
   param sharding use FSDP+DTensor. Make the trade-off explicit in 03.
5. **Ray Train V2.** `RAY_TRAIN_V2_ENABLED=1` as process env AND in worker
   `runtime_env` `env_vars` (set before importing `ray.train`).
6. **Dependency propagation (verified).** On this Anyscale workspace, a plain
   `pip install pkg==ver` (no flags) **registers the package on all cluster
   nodes** — confirmed: a GPU worker imports torch/transformers/datasets/
   deepspeed/accelerate after a plain install. A `--index-url` install does NOT
   propagate (head-only) — Anyscale warns. So: install with plain pip, no
   `--index-url`, no uv. `deepspeed==0.18.9` builds cleanly on this image (no
   `PIP_NO_BUILD_ISOLATION` needed). Notebooks use
   `runtime_env={"working_dir": ".", "env_vars": {...}}` only (no `pip` key).
7. **`kill_train_worker.py`** relies on `ray.util.state.list_actors` finding
   `RayTrainWorker` actors; confirm the state API is enabled on this cluster and
   the live-kill demo recovers within `max_failures`.
8. **Storage.** Use `/mnt/cluster_storage` for anything workers read/write
   across nodes (checkpoints, profiler HTML). Never write training state to head
   node local FS.
9. **Smoke scale.** Cap steps/epochs so each notebook pre-runs in minutes. State
   the production knob ("set `debug_steps=0` / raise `num_epochs`") inline.

---

## 10. Build order (checklist)

1. [x] Create dir skeleton (`common/`, `images/`). Write `BUILD_PLAN.md`.
2. [x] Write `common/utils.py` (precision select, runtime_env, model builders,
       cluster printer), `common/kill_train_worker.py` (copied + de-suffixed),
       `common/__init__.py`, `requirements.txt`. Full stack installed + verified
       importable on a GPU worker (torch 2.9.1+cu128, Tesla T4, deepspeed 0.18.9).
3. [ ] Write `README.md` + `00_scaling_imperative.ipynb` — lock the arc,
       numbering, cross-reference targets, cluster image.
4. [ ] Build `prerequisite_00_ray_data.ipynb` — restyle, T4-tune, run, commit.
5. [ ] Build `01_ray_foundations.ipynb` — run on 4× T4 smoke config, commit;
       verify checkpoint lands in `/mnt/cluster_storage`.
6. [ ] Build `02_memory_and_sharding.ipynb` — FSDP + DeepSpeed paths, memory
       profile, run, commit.
7. [ ] Build `03_2d_parallelism.ipynb` — DTensor + AutoTP 2×2, run, commit.
8. [ ] Build `04_fault_tolerance_and_observability.ipynb` — live-kill demo +
       profiling, run, commit.
9. [ ] Build `05_putting_it_together_lfm2.ipynb` — synthesis, commit.
10. [ ] Final pass: grep for external refs (§8); house-style consistency (§6);
        all notebooks have committed outputs; no W&B; `git init` + commit.

---

## 11. Time budget (context — informs core-vs-depth, not a hard script)

3-hour morning, one break. Instructor narrates pre-run notebooks, deep-diving a
few modules and touring the rest. Design every section so skipping reads
gracefully.

| Beat | Min | Spine |
|---|---|---|
| Scaling imperative (00) | 18 | narrate |
| Ray foundations (01) | 25 | core |
| Memory & sharding (02) | 35 | core (depth flex) |
| **Break** | 15 | — |
| 2D parallelism (03) | 35 | core |
| Fault tolerance & observability (04) | 30 | core |
| Putting it together + LFM2 (05) | 15 | synthesis |
| Wrap / scaling knobs / Q&A | 7 | — |

(`prerequisite_00` is pre-reading, not presented live.)

---

## 12. Decisions locked (do not relitigate)

- **Notebook-only** deliverable; markdown cells are the slides; no deck, no
  run-of-show.
- **Standalone & self-contained**; cross-reference `prerequisite_00` + `00`–`05`
  only.
- **House style** applied to all notebooks (mirrors robotics repo).
- **Ship clean (no committed outputs)**, but every notebook must execute cleanly
  top-to-bottom at smoke scale on this 4× T4 cluster (verify by running, then
  strip outputs). Changed 2026-06-11 from the original pre-run-with-outputs plan.
- **Ship the Ray Data prereq**; **Ray Train prereq is folded into `01`** (not a
  separate notebook).
- **PP & EP conceptual only** for now (runnable toys are a possible later add).
- **Mixed hero models:** gpt2/small-LLM (01, 02) → Qwen2.5-0.5B (03) → keep
  ResNet/ViT where their memory-profiler / live-kill / profiler demos shine.
- **Megatron is reference-only** (Anyscale Megatron-Bridge tutorial), no
  runnable Megatron code.
- **Hardware:** 4× T4, fp16, PCIe (no NVLink), 4 GPUs constant (1×4 or 2×2),
  image `anyscale/ray:2.55.1-py311-cu128`.

---

## 13. Out of scope

- Runnable pipeline-parallel or expert-parallel/MoE training (conceptual only;
  possible later add).
- Runnable Megatron-LM / Megatron-Bridge (reference link only).
- True multi-node training demos unless the day-of cluster becomes 2 nodes ×
  2 GPUs (code stays topology-agnostic either way).
- The missing `02_TorchFT_Integration.ipynb` (per-step quorum recovery) — source
  not available; mention TorchFT conceptually in 04 at most.
- Attendee-provisioning logistics (handled by Anyscale/AWS).
- Hyperparameter tuning / loss optimization (smoke-scale runs validate
  mechanics, not accuracy).
```
