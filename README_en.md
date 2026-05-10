<div align="center">

<img src="assets/title.jpg" alt="ComfyUI-naginagi-vibe-lab" width="100%">

**English** | [日本語](README.md)

</div>

- [Top](#comfyui-naginagi-vibe-lab)
- [Installation](#installation)
- [Nodes](#-nodes)
  - [Corrector](#corrector)
  - [Scheduler](#scheduler)
- [Tested Environment](#tested-environment)

# ComfyUI-naginagi-vibe-lab

**A slightly experimental, quite weird, and lovingly ponkotsu ComfyUI node collection by naginagi.**

- `Corrector` group — direct Conditioning correction from LLM image-analysis text
- `Scheduler` group — modulate Conditioning / LoRA along denoising steps and iterative loops
- Designed to work with ComfyUI's built-in `TextGenerate` node (Qwen 3.5 etc.)

## Installation

### A) ComfyUI Manager (recommended)

Use **"Install via Git URL"** in ComfyUI Manager:

```
https://github.com/<your-account>/ComfyUI-naginagi-vibe-lab
```

### B) Manual install

```bash
cd <ComfyUI>/custom_nodes/
git clone https://github.com/<your-account>/ComfyUI-naginagi-vibe-lab.git
```

No additional `pip install` needed — all dependencies are bundled with ComfyUI.

## 🍓 Nodes

All nodes appear under `conditioning/vibe-lab/...` in the ComfyUI menu. They share a `naginagi · ` display-name prefix for easy searching.

---

### Corrector

A node group that **projects-and-subtracts the "problem direction"** out of your Conditioning, using LLM-analysis text as the source vector. This suppresses anatomical errors, style breakdowns, and color casts without re-generation or re-training.

- `ConditioningProjection` 🎯
  - Low-level node performing raw vector projection (add / subtract / replace) between two CONDITIONINGs
- `ConditioningCorrector` 🎯🤖
  - All-in-one: LLM text → CLIP encode → projection subtract (core node)
- `ConditioningCorrectorDual` 🎯➕➖
  - Simultaneously correct Positive and Negative — subtract from Pos, add to Neg
- `ConditioningCorrectorInpaint` 🎯🩹
  - Apply correction only inside a mask. Pair with `SetLatentNoiseMask`

See [docs/corrector.md](docs/corrector.md) for details.

---

### Scheduler

A node group that modulates Conditioning / LoRA / iterative processing along denoising steps and refinement loops. Implements the "bold start, gentle finish" non-linear curve technique.

- `ConditioningStepScheduler` ⏱️📊
  - Non-linear strength modulation with 8 curves (`bold_to_refined` / `weak_to_strong` etc.)
- `LoRAScheduleApply` ⏱️🎚️
  - Step-wise LoRA strength modulation
- `IterativeUpscalePlanner` ⏱️🔁
  - Auto-compute iteration count for iterative upscale loops
- `IterativeStepScale` ⏱️📉
  - Overshoot-safe scale-factor calculator (pairs with `UltimateSDUpscale`)
- `IterativeRefineDenoise` ⏱️🎛️
  - Non-linear `denoise` value per iteration
- `AdaptiveHaltCheck` ⏱️🛑
  - ACT-style adaptive halting — stops the loop when LLM finds no more problems

See [docs/scheduler.md](docs/scheduler.md) for details.

---

## Tested Environment

- ComfyUI 0.18.1 / Python 3.13 / PyTorch CUDA 13.0
- RTX 4060 Ti 16GB / Windows
- SDXL (NoobAI-XL v-pred) / Anima preview3 / Flux.2 Klein

## License

MIT License — see [LICENSE](LICENSE).
