from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

from hackathon_app.models import ImageAsset
from hackathon_app.services.mock_models import describe_image


SYSTEM_PROMPT = """あなたは医療者の確認を支援するVision Language Modelです。画像から見える外傷・創傷の外観所見のみを説明してください。診断名、重症度、治療方針、受診要否は断定しないでください。"""

USER_PROMPT = """画像から見える外傷・創傷所見を、以下のJSON形式で整理してください。
診断名、重症度、治療方針、受診要否は断定しないでください。

{
  "外傷分類": "",
  "損傷部位": "",
  "視認所見": "",
  "出血": "",
  "熱傷所見": "",
  "あざ内出血": "",
  "開放創裂創": "",
  "腫脹変形": "",
  "汚染異物": "",
  "緊急度": "",
  "緊急度理由": "",
  "状況説明": ""
}"""


@lru_cache(maxsize=1)
def _load_vlm() -> tuple[Any, Any, str, int, str, str]:
    import torch
    model_id = os.getenv("LFM_VL_MODEL", "LiquidAI/LFM2.5-VL-1.6B")
    _requested = os.getenv("LFM_VL_DEVICE", os.getenv("LFM_AUDIO_DEVICE", "cuda:0"))
    device = _requested if torch.cuda.is_available() else "cpu"
    if device != _requested:
        print("VLM: CUDA unavailable, falling back to CPU")
    max_new_tokens = int(os.getenv("LFM_VL_MAX_NEW_TOKENS", "256"))
    system_prompt = os.getenv("LFM_VL_SYSTEM_PROMPT", SYSTEM_PROMPT)
    user_prompt = os.getenv("LFM_VL_USER_PROMPT", USER_PROMPT)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto" if device.startswith("cuda") else None,
        dtype=dtype,
        trust_remote_code=True,
    ).eval()
    if not device.startswith("cuda"):
        model.to(device)
    return model, processor, device, max_new_tokens, system_prompt, user_prompt


def analyze_image(image: ImageAsset) -> str:
    if os.getenv("LFM_VL_ENABLED", "1").lower() in {"0", "false", "no"}:
        return describe_image(image)

    try:
        return _analyze_image_with_lfm_vl(Path(image.path))
    except Exception as exc:
        print(f"LFM VL fallback: {exc}", flush=True)
        fallback = describe_image(image)
        if os.getenv("LFM_VL_SHOW_FALLBACK_REASON", "0").lower() in {"1", "true", "yes"}:
            return f"{fallback}（VLM推論は失敗: {exc}）"
        return fallback


def _analyze_image_with_lfm_vl(path: Path) -> str:
    import json
    model, processor, _device, max_new_tokens, system_prompt, user_prompt = _load_vlm()
    image = load_image(str(path))
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    generated = outputs[:, input_len:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return _format_vlm_json(raw)


def _format_vlm_json(raw: str) -> str:
    import json, re
    raw = raw.strip()
    # JSON ブロックを抽出
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return raw or "画像所見を取得できませんでした。"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return raw
    # 空でないフィールドだけ表示
    lines = []
    for key, val in data.items():
        if val and str(val).strip() not in ("", "なし", "不明"):
            lines.append(f"【{key}】{val}")
    return "\n".join(lines) if lines else raw
