# Training and Dataset Notes

## 1. Purpose

The ASR experiment trains a Japanese role-tagged ASR model for disaster-medical conversations.

The target output uses SOT-style special prefix tokens:

```text
<doctor> 医師発話
<patient> 患者発話
```

The main goal is to reduce speaker-role flips, especially when a chunk contains a doctor/patient turn exchange.

## 2. Datasets

### 2.1 BeTraC-Derived Data

BeTraC-derived data was used as an earlier validation path for role-tagged ASR.

High-level pipeline:

1. Start from BeTraC medical consultation text/audio data.
2. Translate or adapt text into Japanese where needed.
3. Generate Japanese speech with VOICEVOX.
4. Preserve turn timestamps from TTS.
5. Build role-tagged ASR targets.
6. Evaluate ASR quality and role-label accuracy.

Notes:

- BeTraC-derived synthetic Japanese data is useful as a secondary validation set.
- It is not the primary dataset for the disaster-medicine track because its conversation style and topic distribution differ from disaster triage.
- Translation/TTS artifacts can make metrics noisy, so BeTraC should be treated as an external-distribution check.

### 2.2 Synthetic Disaster-Triage Data

The main training dataset is synthetic disaster-medical dialogue data.

Raw format:

```json
{"text": "<doctor> ...\n<patient> ..."}
```

Cleaned format:

```json
{
  "id": "gousei_00001",
  "triage_label": "green",
  "text": "<doctor> ...\n<patient> ...",
  "turns": [
    {"role": "doctor", "text": "..."},
    {"role": "patient", "text": "..."}
  ],
  "turn_count": 6,
  "doctor_turn_count": 4,
  "patient_turn_count": 2
}
```

Triage labels:

```text
green / yellow / red / black / unknown
```

## 3. TTS Data Creation

VOICEVOX is used to synthesize the cleaned text conversations.

Rules:

- Assign one doctor voice and one patient voice per dialogue.
- Randomize voices across dialogues.
- Keep voice assignments fixed within a dialogue.
- Preserve each turn's start/end timestamp.
- Save speaker ids and speaker names.

Example command:

```bash
python scripts/synthesize_gousei_with_voicevox.py \
  --input data/gousei/dataset_clean.jsonl \
  --output-audio-root data/gousei/voicevox_audio_random_safe_timed \
  --output-manifest data/manifests/gousei_voicevox_random_safe_timed.jsonl \
  --engine-url http://127.0.0.1:50021 \
  --randomize-speakers \
  --speaker-pool 8,10,11,12,14,16,21,23,47,52,53,67,69,74,99,100,107,108,109,118,122 \
  --seed 23
```

The output manifest keeps `turns[]` with timestamps.

## 4. Chunking Method

Training chunks are created from TTS turn timestamps, not VAD.

Do not split inside a turn.

Target chunk mix:

```text
single-turn          40%
two-turn exchange    35%
multi-turn mixed     20%
longer mixed          5%
```

Chunk types:

- `single`: exactly one turn
- `two_turn`: exactly two turns, preferably one doctor/patient exchange
- `multi_turn`: 3 to 5 turns with both roles
- `longer_mixed`: longer mixed-role chunks up to about 24 seconds

This mix intentionally increases chunks with speaker changes to train the model against role flips.

Example command:

```bash
python scripts/build_gousei_sot_ratio_chunks.py \
  --manifest data/manifests/gousei_voicevox_random_safe_timed.jsonl \
  --output-audio-root data/gousei/sot_ratio_chunks/audio \
  --output-manifest data/manifests/gousei_sot_ratio_chunks.jsonl \
  --chunks-per-source 10 \
  --single-ratio 0.40 \
  --two-turn-ratio 0.35 \
  --multi-turn-ratio 0.20 \
  --longer-ratio 0.05 \
  --seed 29
```

## 5. Train/Validation Split

Split by `source_id`, not by chunk.

This prevents chunks from the same source dialogue from appearing in both train and validation sets.

Example:

```bash
python scripts/split_manifest_by_source.py \
  --manifest data/manifests/gousei_sot_ratio_chunks.jsonl \
  --train-output data/manifests/gousei_sot_ratio_chunks_train.jsonl \
  --val-output data/manifests/gousei_sot_ratio_chunks_val.jsonl \
  --val-ratio 0.1 \
  --seed 31
```

## 6. Preprocessing

Preprocess manifests for LFM2.5-Audio-JP.

Special tokens are added before mapping:

```text
<doctor>, <patient>
```

Example:

```bash
python scripts/preprocess_asr_manifest.py \
  --manifest data/manifests/gousei_sot_ratio_chunks_train.jsonl \
  --output data/processed/gousei_sot_ratio_chunks_train_lfm25jp \
  --model LiquidAI/LFM2.5-Audio-1.5B-JP \
  --device cuda:0 \
  --system-prompt 'Perform ASR in japanese. Use <doctor> and <patient> speaker prefix tokens.' \
  --special-tokens '<doctor>,<patient>' \
  --max-context-length 4096 \
  --overwrite
```

Repeat for the validation manifest.

## 7. Training Method

Base model:

```text
LiquidAI/LFM2.5-Audio-1.5B-JP
```

Trainable components:

- audio adapter
- LoRA modules
- added special-token embedding rows

The tokenizer is extended with special tokens, but the tokenizer itself is not trained.

Recommended hyperparameters:

```text
LoRA rank       16
LoRA alpha      16
LoRA dropout    0.05
LR              1e-4
weight decay    0.01
scheduler       cosine
warmup steps    150
batch size      4
grad accum      4
effective batch 16
max steps       4000
```

Example:

```bash
python scripts/train_lfm2_audio_asr_adapter_lora.py \
  --train-data data/processed/gousei_sot_ratio_chunks_train_lfm25jp \
  --eval-data data/processed/gousei_sot_ratio_chunks_val_lfm25jp \
  --model LiquidAI/LFM2.5-Audio-1.5B-JP \
  --output-dir checkpoints/gousei/lfm25_audio_jp_sot_ratio_r16_lr1e4_cosine_b4a4_steps4000 \
  --device cuda:0 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --max-steps 4000 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --lr-scheduler cosine \
  --warmup-steps 150 \
  --max-grad-norm 1.0 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --train-special-token-embeddings \
  --eval-interval 250 \
  --eval-max-batches 0 \
  --save-best \
  --save-interval 500 \
  --log-interval 20
```

## 8. Evaluation

Evaluate both ASR quality and role-label quality.

Metrics:

- CER
- WER
- first label accuracy
- label sequence prefix accuracy
- label sequence exact accuracy

Important buckets:

- all
- single-turn
- two-turn exchange
- multi-turn mixed
- both-roles chunks
- starts-with-doctor
- starts-with-patient

The most important role-flip metrics are:

```text
both_roles exact
two_turn exact
starts_with_patient first_label_acc
```

Example inference:

```bash
python scripts/asr_lfm2_audio_with_adapter.py \
  --manifest data/manifests/gousei_sot_ratio_chunks_val.jsonl \
  --model LiquidAI/LFM2.5-Audio-1.5B-JP \
  --checkpoint checkpoints/gousei/lfm25_audio_jp_sot_ratio_r16_lr1e4_cosine_b4a4_steps4000/adapter_lora_best.pt \
  --device cuda:0 \
  --max-new-tokens 180 \
  --system-prompt 'Perform ASR in japanese. Use <doctor> and <patient> speaker prefix tokens.' \
  --quiet \
  --output checkpoints/gousei/asr_val_best.jsonl
```

Example evaluation:

```bash
python scripts/eval_role_tag_asr.py \
  --pred checkpoints/gousei/asr_val_best.jsonl
```
