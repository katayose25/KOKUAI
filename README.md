# KokuAI Disaster Medicine Demo

KokuAI is a hackathon prototype for disaster-medical triage support. It provides a local web application for audio transcription, clinical prompt extraction, chart drafting, and rule-based triage support.

This repository snapshot is the public/submission package. Large generated datasets, model checkpoints, logs, W&B runs, and local secrets are intentionally excluded.

## 1. Run the Demo App

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-app.txt
```

For ASR training, evaluation, and data generation scripts, also install:

```bash
pip install -r requirements-training.txt
```

If you use CUDA, install PyTorch wheels matching your CUDA/runtime environment before or during these installs.

### Configure

Copy the example env file:

```bash
cp hackathon_app/.env.example hackathon_app/.env
```

Edit `hackathon_app/.env` and set local model/checkpoint paths as needed.

Important: do not commit `.env`.

Model adapters are loaded from Hugging Face by default:

```text
LFM_AUDIO_CHECKPOINT_REPO=kinohito/koku-ai-asr-lora-jp
LFM_CHART_LORA_REPO=kinohito/koku-ai-transcript-lora
```

If the repositories are private, set `HF_TOKEN` or run `huggingface-cli login` before starting the app.

### Start

```bash
python -m uvicorn hackathon_app.main:app --host 0.0.0.0 --port 8888
```

Open:

```text
http://localhost:8888
```

or, on a remote server:

```text
http://<server-ip>:8888
```

## 2. Repository Contents

```text
hackathon_app/   FastAPI + Jinja2 + HTMX demo application
scripts/         ASR data preparation, TTS, chunking, training, inference, evaluation
docs/            Dataset and training documentation
```

## 3. Included Training/Inference Scripts

Core scripts:

- `scripts/synthesize_gousei_with_voicevox.py`
- `scripts/build_gousei_sot_ratio_chunks.py`
- `scripts/split_manifest_by_source.py`
- `scripts/preprocess_asr_manifest.py`
- `scripts/train_lfm2_audio_asr_adapter_lora.py`
- `scripts/asr_lfm2_audio_with_adapter.py`
- `scripts/eval_role_tag_asr.py`
- `scripts/eval_role_tag_asr_stratified.py`

A few helper scripts are included because the core scripts import shared utility functions from them.

## 4. Training Documentation

See:

```text
docs/training_and_datasets.md
```

It describes:

- BeTraC-derived data
- synthetic disaster-triage data
- VOICEVOX TTS generation
- ratio-controlled SOT chunking
- special-token LoRA training
- evaluation metrics

## 5. What Is Not Included

The following are intentionally excluded:

- `.env`
- generated audio files
- processed datasets
- model checkpoints
- W&B run directories
- logs
- API keys or local server URLs

## 6. Public Release Notes

Before publishing generated data or model checkpoints, confirm licenses for:

- upstream datasets
- TTS voices
- base models
- generated synthetic data
- trained adapter checkpoints
