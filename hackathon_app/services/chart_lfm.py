from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from hackathon_app.models import ChartDraft, ClinicalPrompt, Encounter, Patient, TranscriptTurn


class ChartGenerationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_chart_model() -> tuple[Any, Any, str, int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = os.getenv("LFM_CHART_MODEL", "LiquidAI/LFM2-2.6B-Transcript")
    device = os.getenv("LFM_CHART_DEVICE", os.getenv("LFM_AUDIO_DEVICE", "cuda:0"))
    max_new_tokens = int(os.getenv("LFM_CHART_MAX_NEW_TOKENS", "700"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).eval()

    lora_repo = os.getenv("LFM_CHART_LORA_REPO", "").strip()
    if lora_repo:
        try:
            from peft import PeftModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("peft is required when LFM_CHART_LORA_REPO is set") from exc
        revision = os.getenv("LFM_CHART_LORA_REVISION", "").strip() or None
        model = PeftModel.from_pretrained(model, lora_repo, revision=revision).eval()

    model.to(device)
    return tokenizer, model, device, max_new_tokens


def generate_soap_with_lfm(
    patient: Patient,
    encounter: Encounter,
    turns: list[TranscriptTurn],
    prompts: list[ClinicalPrompt],
) -> ChartDraft:
    if os.getenv("LFM_CHART_ENABLED", "1") in {"0", "false", "False", "no"}:
        raise ChartGenerationError("LFM chart generation disabled")

    tokenizer, model, device, max_new_tokens = _load_chart_model()
    prompt = _build_prompt(patient, encounter, turns, prompts)
    inputs = _encode_prompt(tokenizer, prompt, device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    input_len = inputs["input_ids"].shape[-1]
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    data = _extract_json(generated)
    return ChartDraft(
        subjective=_field_text(data.get("subjective", "")),
        objective=_field_text(data.get("objective", "")),
        assessment=_field_text(data.get("assessment", "")),
        plan=_field_text(data.get("plan", "")),
        handoff=_field_text(data.get("handoff", "")),
    )



def _field_text(value: Any) -> str:
    """Normalize model JSON fields into displayable Japanese text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "note" in value and len(value) == 1:
            return str(value["note"]).strip()
        parts = []
        for key, item in value.items():
            text = _field_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "。".join(parts).strip()
    if isinstance(value, list):
        parts = [_field_text(item) for item in value]
        return "。".join(part for part in parts if part).strip()
    return str(value).strip()

def _encode_prompt(tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "あなたは医療記録の専門家です。"
                "患者と医師の逐語録を分析し、"
                "subjective・objective・assessment・plan・metadata の各フィールドを含む"
                "SOAP カルテを JSON 形式で出力してください。"
                "余分なテキストは含めず、JSON のみを返してください。"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
    else:
        text = prompt
    encoded = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _build_prompt(
    patient: Patient,
    encounter: Encounter,
    turns: list[TranscriptTurn],
    prompts: list[ClinicalPrompt],
) -> str:
    transcript = "\n".join(_format_turn(turn) for turn in turns)
    prompt_lines = "\n".join(f"- {p.kind}: {p.title} - {p.detail}" for p in prompts) or "- none"
    image_lines = "\n".join(f"- {image.finding}" for image in encounter.images if image.finding) or "- none"

    return f"""
以下の診察会話の文字起こしから、日本語のSOAP形式カルテ草稿を作成してください。
話者同定が未完了の場合があります。speaker=unknownでも、内容からS/O/A/Pを整理してください。
事実として書かれていない情報は推測で断定しないでください。不明点は plan または handoff に確認事項として書いてください。

患者情報:
- 年齢: {patient.age}
- 性別: {patient.sex}
- 主訴: {patient.chief_complaint or "未入力"}
- メモ: {patient.memo or "なし"}

Clinical Prompts:
{prompt_lines}

画像/添付所見:
{image_lines}

Transcript:
{transcript}

以下のJSONだけを返してください。Markdownや説明文は不要です。
{{
  "subjective": "Sを日本語で。患者の訴え、経過、陰性所見を簡潔に。",
  "objective": "Oを日本語で。バイタル、身体所見、画像所見、確認済み客観所見。不明なら未確認と書く。",
  "assessment": "Aを日本語で。鑑別・評価。断定しすぎない。",
  "plan": "Pを日本語で。追加問診、診察、検査、処置、説明。",
  "handoff": "未確認事項や次に確認すべきこと。"
}}
""".strip()


def _format_turn(turn: TranscriptTurn) -> str:
    start = "" if turn.start is None else f"{turn.start:.1f}-{(turn.end or turn.start):.1f}s "
    return f"{start}{turn.speaker}: {turn.text}"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise ChartGenerationError(f"No JSON object found in chart model output: {text[:300]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ChartGenerationError("Chart model output was not a JSON object")
    return data
