<div align="center">

# Global Transport Couplings for Classifier-Free Guided Flows

[![Paper](http://img.shields.io/badge/paper-arxiv.2610.07555-B31B1B.svg)](https://arxiv.org/abs/2610.07555)
[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![accelerate](https://img.shields.io/badge/🤗_Accelerate-multi--GPU-yellow)](https://huggingface.co/docs/accelerate)
[![uv](https://img.shields.io/badge/env-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/kpetrovicc/global-transport/blob/main/LICENSE)<br>

<p float="left">
  <img align="top" width="46%" src="assets/guidance_sweep.gif" style="display: inline-block; margin: 0 auto; max-width: 420px"/>
  <img align="top" width="46%" src="assets/samples.png" style="display: inline-block; margin: 0 auto; max-width: 420px"/>
</p>

<sub>Left: the same noise decoded by SiT-XL/2 trained with the independent coupling and with Global Transport while the guidance scale sweeps from <i>w</i> = 1 to 4. Right: ImageNet 256×256 samples.</sub>

</div>

## Description

**Global Transport (GT)** is a class-agnostic optimal-transport coupling for training conditional flow models. Optimal-transport couplings are known to reduce training variance in unconditional flows, but their role in conditional generation was unclear: building a separate coupling per condition is impractical for large or continuous conditioning spaces. GT instead solves one exact transport plan between the noise and data of a training batch, without looking at the labels. On its own this *hurts* unguided generation, since different conditions get associated with different regions of the source noise. Combined with **classifier-free guidance (CFG)**, however, GT consistently improves generation across domains, model scales and sampling budgets. The paper argues that couplings for conditional flows should therefore be judged under the guided flow used at inference, and identifies coupling design as a training-time axis that improves guided generation without touching architectures, samplers or guidance mechanisms.

This repository trains and evaluates **SiT** flow models and **Decoupled MeanFlow (DMF)** flow maps on ImageNet 256×256 with three ways of pairing noise with data during training:

| coupling | flag | what is paired |
|---|---|---|
| independent | (none) | independent noise, i.e. standard flow matching |
| Global Transport | `--use-ot` | the noise batch is permuted by the exact squared-Euclidean assignment to the data batch, ignoring labels |
| class-conditional OT | `--use-ccot` | the same assignment, restricted to noise–data pairs of the same class ([C2OT](https://arxiv.org/abs/2503.10636)) |

Under multi-GPU training both transport plans are solved over the **global** batch (the data and noise of all processes are all-gathered, rank 0 solves the assignment and broadcasts it), so the coupling is set by `--batch-size` alone and does not change with the number of GPUs. Besides class labels, the flow models can be conditioned on **continuous conditions** (caption embeddings). A self-contained 2-D example of all three couplings under CFG lives in [examples/fab4_notebook.ipynb](examples/fab4_notebook.ipynb).

The code is a fork of [Decoupled MeanFlow](https://github.com/kyungmnlee/DMF), which builds on [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA) and [edm2](https://github.com/NVlabs/edm2); the OT plan sampler comes from [torchcfm](https://github.com/atong01/conditional-flow-matching). See [the paper](https://arxiv.org/abs/2610.07555) for details.

### Data

All scripts read ImageNet 256×256 in the edm2 layout through `--data-dir` (`DATA_DIR` in the scripts):

```
DATA_DIR/
  vae-sd/   00000/img-mean-std-00000000.npy ...  dataset.json    # sd-vae-ft-mse latents + labels (training)
  images/   00000/img00000000.png ...            dataset.json    # 256x256 images (only for REPA and caption checks)
```

* **ImageNet-256 latents and images**: we recommend to download the already converted dataset (~295 GB of archives from [Yuehao/imagenet-1k-256](https://huggingface.co/datasets/Yuehao/imagenet-1k-256)) with [scripts/data/download_imagenet256.sh](scripts/data/download_imagenet256.sh) after setting `DATA_DIR` in the script, otherwise one can also convert ImageNet following the [preprocessing guide](preprocessing/README.md) as mentioned DMF repo.
* **Reference statistics for FID / FD-DINOv2** (`imgnet256.pkl`, 42 MB) from the [DMF Huggingface repository](https://huggingface.co/kyungmnlee/DMF), downloaded once into `ckpt/` where the evaluation scripts expect them:
  ```bash
  pip install huggingface_hub
  hf download kyungmnlee/DMF imgnet256.pkl --local-dir ckpt
  ```
* **Caption embeddings** for the continuous-condition experiments (the caption setup of C2OT): every training image gets a caption from [visual-layer/imagenet-1k-vl-enriched](https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched), encoded with the DFN5B-CLIP-ViT-H-14 text encoder into a 1024-d unit vector. Set `DATA_DIR` and `COND_DIR` in [scripts/data/text_conditions_imagenet256.sh](scripts/data/text_conditions_imagenet256.sh) and run it (needs `pip install open_clip_torch pyarrow`). It writes `train_text_emb_dfn5b.npy` (one row per training latent, in the order of `vae-sd/`) and `val_text_emb_dfn5b.npy` (the 50,000 validation captions, used for sampling), and checks the train row order against the labels in `dataset.json` and, when the images are available, against image-text similarity.

💡 The scripts carry `PATH/TO/...` placeholders for the data, condition, checkpoint and output directories; fill them in before launching.

If you find this code useful in your research, please cite our work.

<summary>
K. Petrović, Z. W. Blasingame, D. Rehman, İ. İ. Ceylan, M. Bronstein, S. Y. Zhang, L. Atanackovic, A. Tong. "Global Transport Couplings for Classifier-Free Guided Flows." <em>arXiv:2610.07555</em>, 2026.
</summary>

```bibtex
@article{petrovic2026global,
  title={Global Transport Couplings for Classifier-Free Guided Flows},
  author={Katarina Petrovi{\'c} and Zander W. Blasingame and Danyal Rehman and {\.I}smail {\.I}lkan Ceylan and Michael Bronstein and Stephen Y. Zhang and Lazar Atanackovic and Alexander Tong},
  journal={arXiv preprint arXiv:2610.07555},
  year={2026},
  url={https://arxiv.org/abs/2610.07555}
}
```

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/kpetrovicc/global-transport.git
cd global-transport

# [OPTIONAL] create conda environment
conda create -n global-transport python=3.12 -y
conda activate global-transport

# install pytorch according to instructions
# https://pytorch.org/get-started/
pip install torch==2.8.0 torchvision==0.23.0

# install requirements
pip install -r requirements.txt
```

The launch scripts live in [scripts/data/](scripts/data/) (dataset and caption preparation), [scripts/train/](scripts/train/) (training scripts) and [scripts/eval/](scripts/eval/) (computing FID and FD-DINOv2 across CFG scales). Run them from the repository root, e.g. `sh scripts/train/train_sit_b_2_256_ot.sh`.

One script per model scale, each training with the Global Transport coupling: [scripts/train/train_sit_b_2_256_ot.sh](scripts/train/train_sit_b_2_256_ot.sh), [scripts/train/train_sit_l_2_256_ot.sh](scripts/train/train_sit_l_2_256_ot.sh) and [scripts/train/train_sit_xl_2_256_ot.sh](scripts/train/train_sit_xl_2_256_ot.sh). Fill in `DATA_DIR`, `OUTPUT_DIR` and the W&B settings at the top of the script. All SiT runs (B/2, L/2, XL/2) train for 800k steps at a batch size of 256, without REPA. We trained our models on a single GPU. 

```bash
accelerate launch train_fm.py \
  --model "SiT-B/2" \
  --attn-func "fa3" \
  --mixed-precision "bf16" \
  --P-mean 0.0 \
  --proj-coeff 0 \
  --max-train-steps 800000 \
  --use-ot
```

Checkpoints are written every 50k steps to `OUTPUT_DIR/EXP_NAME/checkpoints/<step>.pt` (EMA and raw weights). To change the setup, edit the flags in the script:

- **Coupling**: remove `--use-ot` for the independent coupling, or replace it by `--use-ccot` for class-conditional OT.
- **Batch size and GPUs**: `--batch-size` (default 256) is the global batch and hence the size of the transport plan; `--num_processes` of `accelerate launch` sets the number of GPUs it is split over, so changing the GPU count does not change the coupling. Set `--num_processes 1` and drop `--multi_gpu` for a single GPU.
- `--model`: `[SiT-B/2, SiT-L/2, SiT-XL/2]`
- `--attn-func`: `["fa3", "fa2", "torch_sdpa", "base"]`
- `--proj-coeff`: `> 0` enables REPA (needs `DATA_DIR/images` and the DINOv2 weights via `torch.hub`), `0` for standard training
- `--cond-file`, `--cond-dim`: continuous conditions instead of class labels (see below)
- `--max-train-steps`, `--checkpointing-steps`, `--learning-rate`, `--cfg-prob`: training length, checkpoint interval (50k), learning rate (1e-4) and class-dropout probability (0.1)
- `--resume-step`: resume from `OUTPUT_DIR/EXP_NAME/checkpoints/<step>.pt`

### Train flow maps (DMF)

We use DMF to fine-tune pretrained SiT flow matching models into a flow map to enable one- to few-step generation. Start from the SiT checkpoint (`PRETRAINED_CKPT` in the script), see [scripts/train/train_dmf_b_2_256_ot.sh](scripts/train/train_dmf_b_2_256_ot.sh), [scripts/train/train_dmf_l_2_256_ot.sh](scripts/train/train_dmf_l_2_256_ot.sh) and [scripts/train/train_dmf_xl_2_256_ot.sh](scripts/train/train_dmf_xl_2_256_ot.sh). DMF-B/2 run trains for 400k steps at global batch 256 on 1 GPU with model guidance. We train DMF-L/2 model on GPU and with gradient accumulation. 

```bash
accelerate launch train_dmf.py \
  --model "DMFT-B/2" \
  --pretrained PATH/TO/SIT-B-2-256/CKPT \
  --attn-func "fa3" \
  --dmf-depth 8 \
  --g-type "mg" \
  --omega 0.5 \
  --g-min 0.0 \
  --g-max 1.0 \
  --P-mean 0.0 \
  --P-mean-t 0.4 \
  --P-mean-r -1.2 \
  --use-ot
```

| model | `--dmf-depth` | `--omega` | `--g-min` / `--g-max` |
|---|---|---|---|
| `DMFT-B/2` | 8 | 0.5 | 0.0 / 1.0 |
| `DMFT-L/2` | 18 | 0.6 | 0.0 / 0.7 |
| `DMFT-XL/2` | 20 | 0.6 | 0.0 / 0.7 |

- **Coupling**: `--use-ot` couples both branches of the DMF loss (flow matching and MeanFlow). Remove it for the independent coupling, or replace it by `--use-ccot` for class-conditional OT. `--batch-size` and `--num_processes` work as for the flow models.
- `--g-type`: `["default", "mg", "distill"]`. `mg` stands for training with model guidance, `default` for non-cfg, and `distill` for cfg velocity from the teacher flow model.
- `--dmf-depth`: depth of the DMF model (usually 2/3 of the total transformer depth)
- `--omega`: model guidance scale; `--g-min`, `--g-max`: guidance interval for `mg`
- `--P-mean`, `--P-mean-t`, `--P-mean-r`: time proposal distributions
- `--qk-norm`: stabilises DMF training at higher resolution (unstable JVP computation)

### Train with continuous conditions

[scripts/train/train_sit_cont_b_2_256_ot.sh](scripts/train/train_sit_cont_b_2_256_ot.sh) trains SiT-B/2 on the caption embeddings instead of class labels (`--cond-file COND_FILE --cond-dim 1024`); classifier-free guidance drops the condition to the zero vector. Remove `--use-ot` for the independent coupling. The other parameters are those of the flow models.

To sample from such a model, pass the validation caption embeddings to `generate.py` (`--cond-file PATH/TO/val_text_emb_dfn5b.npy --cond-dim 1024`): image `k` is generated from row `k % 50000` of the file, so the png index identifies its caption.

### Evaluate

One script per model, each a plain classifier-free-guidance sweep that samples 50k images per configuration with `generate.py` (EMA weights, Euler sampler, full guidance interval) and scores them with `preprocessing/calculate_metrics.py` (FID and FD-DINOv2 against `ckpt/imgnet256.pkl`). The scripts resolve the repository root from their own location, write the samples to `samples/`, and only need `CKPT_DIR` at the top pointed at your checkpoint:

- [scripts/eval/eval_sit_b_2_256.sh](scripts/eval/eval_sit_b_2_256.sh), [scripts/eval/eval_sit_l_2_256.sh](scripts/eval/eval_sit_l_2_256.sh), [scripts/eval/eval_sit_xl_2_256.sh](scripts/eval/eval_sit_xl_2_256.sh): flow models, 16 / 32 / 64 Euler steps × CFG scales 1.0-4.0
- [scripts/eval/eval_dmf_b_2_256.sh](scripts/eval/eval_dmf_b_2_256.sh), [scripts/eval/eval_dmf_l_2_256.sh](scripts/eval/eval_dmf_l_2_256.sh), [scripts/eval/eval_dmf_xl_2_256.sh](scripts/eval/eval_dmf_xl_2_256.sh): flow maps, 1 / 2 / 4 Euler steps, no CFG (the guidance is trained in)
- [scripts/eval/eval_sit_cont_b_2_256.sh](scripts/eval/eval_sit_cont_b_2_256.sh): the continuous-condition model, 64 Euler steps × CFG scales 1.0-10.0, conditioned on the validation caption embeddings (`COND_FILE`)

The same coupling-agnostic scripts evaluate independent, Global Transport and class-conditional OT checkpoints; only `CKPT_DIR` changes. The commands for SiT model are:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model "SiT-B/2" \
  --ckpt YOUR_CHECKPOINT_PATH \
  --sample-dir SAMPLE_DIR \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode "euler" \
  --num-steps 64 \
  --cfg-scale 2.0 \
  --g-min 0.0 \
  --g-max 1.0 \
  --global-seed 3
```

Flow maps are sampled with 1, 2 or 4 steps. The command for DMF model is:

```bash
torchrun --nnodes=1 --nproc_per_node=8 generate.py \
  --model "DMFT-B/2" \
  --dmf-depth 8 \
  --ckpt YOUR_CHECKPOINT_PATH \
  --sample-dir SAMPLE_DIR \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode "euler" \
  --num-steps 4 \
  --global-seed 3
```

- `--mode`: `["euler", "restart"]`; `--heun` uses two model evaluations per step
- `--cfg-scale`: classifier-free guidance scale (`1.0` = no guidance)
- `--g-min`, `--g-max`: guidance interval, CFG is applied only for `t` in `[g-min, g-max]` (`t = 1` is pure noise; default: the full interval). `--g-max 0.7` is the tuned interval used alongside the full interval in the paper; the sample folder then ends in `gmax-0.7`
- `--dmf-depth`: required for DMF checkpoints (see the table above)
- `--cond-file`, `--cond-dim`: conditions for continuous-condition models

Samples land in `SAMPLE_DIR/<model>-seed-<seed>-mode-euler-steps-<n>-shift-1.0-cfg-<w>-gmin-<a>-gmax-<b>/`; the scripts rebuild this name for the metrics, so write scales and interval bounds with a decimal point (`1.0`, not `1`). FID and FD-DINOv2 are then computed with

```bash
cd preprocessing
python calculate_metrics.py calc \
  --images "SAMPLE_DIR/<folder>" \
  --ref="../ckpt/imgnet256.pkl"
```

### Toy example: couplings and guidance on a 40-mode Gaussian mixture

[examples/fab4_notebook.ipynb](examples/fab4_notebook.ipynb) trains small conditional flow models on a 2-D 40-mode mixture with the three couplings, using the same `OTPlanSampler` as the ImageNet trainers, and plots samples across guidance scales, sampling trajectories, velocity-field heatmaps, divergence maps and the guidance-norm integral. It runs on CPU in a few minutes.

## Contributions

Have a question? Found a bug? Missing a specific feature? Feel free to file a new issue, discussion or PR with respective title and description.

Before making an issue, please verify that:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions for improvements are always welcome!

## Acknowledgement

This code is mainly built upon [DMF](https://github.com/kyungmnlee/DMF), [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA), [DiT](https://github.com/facebookresearch/DiT), [edm2](https://github.com/NVlabs/edm2), [torchcfm](https://github.com/atong01/conditional-flow-matching) and [C2OT](https://arxiv.org/abs/2503.10636).
