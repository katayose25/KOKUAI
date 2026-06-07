from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from hackathon_app.models import AudioAsset, ChartDraft, EncounterStatus, ImageAsset, now_iso
from hackathon_app.services.asr import normalize_transcript_turns, transcribe_chunk_token_stream
from hackathon_app.services.clinical_pipeline import ClinicalTriggerEngine, ConversationBuffer, draft_soap, merge_accumulated_prompts
from hackathon_app.services.vision import analyze_image
from hackathon_app.store import store

router = APIRouter()
templates = Jinja2Templates(directory="hackathon_app/templates")

SPEAKER_LABELS = {"doctor": "医師", "patient": "患者", "speaker_00": "話者00", "speaker_01": "話者01", "unknown": "不明"}
PROMPT_LABELS = {"alert": "ALERT", "negation": "NEGATION", "need_check": "NEED_CHECK", "follow_up": "FOLLOW_UP"}


def _triage_snapshot(encounter) -> dict:
    chart_text = " ".join([
        encounter.chart.subjective,
        encounter.chart.objective,
        encounter.chart.assessment,
        encounter.chart.plan,
        encounter.chart.handoff,
    ]).strip()
    transcript_text = " ".join(turn.text for turn in encounter.transcript)
    prompt_text = " ".join(f"{prompt.title} {prompt.detail}" for prompt in encounter.clinical_prompts)
    text = f"{chart_text} {transcript_text} {prompt_text}".lower()

    red_terms = (
        "搬送", "意識", "ぼんやり", "呼吸困難", "息苦", "ショック", "大量出血", "止まらない",
        "一酸化炭素", "気道", "胸痛", "冷や汗", "しびれ", "麻痺", "チアノーゼ",
    )
    yellow_terms = (
        "骨折", "強い痛み", "歩け", "歩行", "腫れ", "やけど", "喘息", "頭痛", "吐き気", "めまい",
        "脱水", "発熱", "創", "出血",
    )
    green_terms = ("軽い", "歩けます", "しびれはありません", "大丈夫", "様子", "処置します")

    red_hits = [term for term in red_terms if term in text]
    yellow_hits = [term for term in yellow_terms if term in text]
    green_hits = [term for term in green_terms if term in text]

    if red_hits:
        level = "RED"
        label = "最優先"
        tone = "red"
        reason = "生命危機につながる可能性がある症状・処置方針を検出"
    elif yellow_hits:
        level = "YELLOW"
        label = "待機的優先"
        tone = "yellow"
        reason = "観察・処置を要する症状を検出"
    elif chart_text or transcript_text:
        level = "GREEN"
        label = "軽症"
        tone = "green"
        reason = "緊急所見は未検出。歩行可否など追加確認が必要"
    else:
        level = "未判定"
        label = "カルテ作成前"
        tone = "pending"
        reason = ""

    checks = [
        {"name": "歩行", "value": "未確認", "flag": "unknown"},
        {"name": "呼吸", "value": "未確認", "flag": "unknown"},
        {"name": "循環/出血", "value": "要確認" if any(term in text for term in ("出血", "止まらない", "ショック")) else "未確認", "flag": "warn" if any(term in text for term in ("出血", "止まらない", "ショック")) else "unknown"},
        {"name": "意識", "value": "要確認" if any(term in text for term in ("意識", "ぼんやり", "混乱")) else "未確認", "flag": "warn" if any(term in text for term in ("意識", "ぼんやり", "混乱")) else "unknown"},
    ]
    return {
        "level": level,
        "label": label,
        "tone": tone,
        "reason": reason,
        "hits": red_hits[:4] or yellow_hits[:4] or green_hits[:4],
        "checks": checks,
    }


def _cache_state(fn) -> str:
    try:
        return "ready" if fn.cache_info().currsize > 0 else "loading"
    except Exception:
        return "loading"


def _latest_encounter_for(patient_id: str):
    encs = store.list_encounters_for_patient(patient_id)
    return encs[0] if encs else None


def _global_patient_context() -> dict:
    patients = store.list_patients()
    all_encounters = {p.id: _latest_encounter_for(p.id) for p in patients}
    all_triage = {
        p.id: _triage_snapshot(enc) if enc else {"level": "未判定", "tone": "pending", "label": "判定待ち", "reason": "", "hits": [], "checks": []}
        for p, enc in ((p, all_encounters[p.id]) for p in patients)
    }
    return {"all_patients": patients, "all_encounters": all_encounters, "all_triage": all_triage}


@router.get("/api/triage-counts")
async def triage_counts() -> JSONResponse:
    patients = store.list_patients()
    counts: dict[str, int] = {"red": 0, "yellow": 0, "green": 0, "black": 0, "pending": 0}
    for p in patients:
        enc = _latest_encounter_for(p.id)
        tone = _triage_snapshot(enc)["tone"] if enc else "pending"
        counts[tone] = counts.get(tone, 0) + 1
    counts["total"] = len(patients)
    return JSONResponse(counts)


@router.get("/api/model-status")
async def model_status() -> JSONResponse:
    from hackathon_app.services.asr import _load_lfm
    from hackathon_app.services.chart_lfm import _load_chart_model
    from hackathon_app.services.vision import _load_vlm

    return JSONResponse({
        "asr": _cache_state(_load_lfm),
        "chart": _cache_state(_load_chart_model),
        "vlm": _cache_state(_load_vlm),
    })


def _context(encounter_id: str, request: Request | None = None) -> dict:
    encounter = store.get_encounter(encounter_id)
    patient = store.get_patient(encounter.patient_id)
    return {
        "request": request,
        "encounter": encounter,
        "patient": patient,
        "encounter_number": store.encounter_number(encounter.id),
        "history": store.list_encounters_for_patient(patient.id),
        "store": store,
        "speaker_labels": SPEAKER_LABELS,
        "prompt_labels": PROMPT_LABELS,
        "triage": _triage_snapshot(encounter),
        "active": "workspace",
        **_global_patient_context(),
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _save_upload(upload: UploadFile, directory: Path) -> tuple[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.filename or "upload.bin").name
    target = directory / f"{now_iso().replace(':', '-')}_{safe_name}"
    target.write_bytes(await upload.read())
    return safe_name, str(target)


@router.get("/encounters/{encounter_id}", response_class=HTMLResponse)
async def workspace(encounter_id: str, request: Request) -> HTMLResponse:
    return templates.TemplateResponse("workspace.html", _context(encounter_id, request))


@router.get("/encounters/{encounter_id}/record")
async def legacy_record_redirect(encounter_id: str) -> RedirectResponse:
    return RedirectResponse(url=f"/encounters/{encounter_id}", status_code=303)


@router.post("/encounters/{encounter_id}/audio", response_class=HTMLResponse)
async def upload_audio(encounter_id: str, request: Request, audio: UploadFile = File(...)) -> HTMLResponse:
    encounter = store.get_encounter(encounter_id)
    filename, path = await _save_upload(audio, store.upload_dir(encounter.id))
    encounter.audio_sources.append(AudioAsset(kind="upload", filename=filename, path=path))
    encounter.status = EncounterStatus.recording
    store.save_encounter(encounter)
    return templates.TemplateResponse("partials/audio_panel.html", _context(encounter_id, request))


@router.delete("/encounters/{encounter_id}/audio/{audio_id}", response_class=HTMLResponse)
async def delete_audio(encounter_id: str, audio_id: str, request: Request) -> HTMLResponse:
    encounter = store.get_encounter(encounter_id)
    audio = next((item for item in encounter.audio_sources if item.id == audio_id), None)
    if audio:
        Path(audio.path).unlink(missing_ok=True)
    encounter.audio_sources = [item for item in encounter.audio_sources if item.id != audio_id]
    store.save_encounter(encounter)
    return templates.TemplateResponse("partials/audio_panel.html", _context(encounter_id, request))


@router.get("/encounters/{encounter_id}/audio/{audio_id}/play")
async def play_audio(encounter_id: str, audio_id: str) -> FileResponse:
    encounter = store.get_encounter(encounter_id)
    audio = next((item for item in encounter.audio_sources if item.id == audio_id), None)
    if audio is None:
        raise HTTPException(status_code=404, detail="audio not found")
    path = Path(audio.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file not found")
    return FileResponse(path, media_type=mimetypes.guess_type(audio.filename)[0] or "audio/wav", filename=audio.filename)


@router.post("/encounters/{encounter_id}/stream/stop")
async def stop_stream(encounter_id: str) -> JSONResponse:
    encounter = store.get_encounter(encounter_id)
    encounter.live_cancel_requested = True
    encounter.live_status = "stopping"
    encounter.live_message = "文字起こしを停止しています..."
    store.save_encounter(encounter)
    return JSONResponse({"ok": True, "message": "停止要求を送信しました。"})


@router.post("/encounters/{encounter_id}/transcript/clear")
async def clear_transcript(encounter_id: str) -> RedirectResponse:
    encounter = store.get_encounter(encounter_id)
    encounter.transcript = []
    encounter.clinical_prompts = []
    encounter.chart = ChartDraft()
    encounter.live_status = "idle"
    encounter.live_message = ""
    encounter.live_processed_chunks = 0
    encounter.live_cancel_requested = False
    for audio in encounter.audio_sources:
        audio.status = "uploaded"
    encounter.status = EncounterStatus.recording if encounter.audio_sources else EncounterStatus.intake
    store.save_encounter(encounter)
    return RedirectResponse(url=f"/encounters/{encounter.id}", status_code=303)


@router.post("/encounters/{encounter_id}/images", response_class=HTMLResponse)
async def upload_image(encounter_id: str, request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    import asyncio
    encounter = store.get_encounter(encounter_id)
    filename, path = await _save_upload(image, store.upload_dir(encounter.id))
    asset = ImageAsset(filename=filename, path=path)
    asset.finding = await asyncio.get_event_loop().run_in_executor(None, analyze_image, asset)
    encounter.images.append(asset)
    encounter.status = EncounterStatus.recording
    store.save_encounter(encounter)
    return templates.TemplateResponse("partials/vlm_panel.html", _context(encounter_id, request))


@router.delete("/encounters/{encounter_id}/images/{image_id}", response_class=HTMLResponse)
async def delete_image(encounter_id: str, image_id: str, request: Request) -> HTMLResponse:
    encounter = store.get_encounter(encounter_id)
    image = next((item for item in encounter.images if item.id == image_id), None)
    if image:
        Path(image.path).unlink(missing_ok=True)
    encounter.images = [item for item in encounter.images if item.id != image_id]
    store.save_encounter(encounter)
    return templates.TemplateResponse("partials/vlm_panel.html", _context(encounter_id, request))


@router.get("/encounters/{encounter_id}/stream")
async def stream_asr(encounter_id: str) -> StreamingResponse:
    async def events():
        import asyncio

        encounter = store.get_encounter(encounter_id)
        if not encounter.audio_sources:
            yield _sse("error", {"message": "音声ファイルが登録されていません。"})
            return

        encounter.transcript = []
        encounter.clinical_prompts = []
        encounter.live_status = "running"
        encounter.live_message = "文字起こしを開始しています..."
        encounter.live_processed_chunks = 0
        encounter.live_cancel_requested = False
        encounter.status = EncounterStatus.recording
        store.save_encounter(encounter)
        yield _sse("status", {"stage": "asr", "message": "文字起こしを開始しています..."})

        try:
            loop = asyncio.get_event_loop()
            turn_iter = transcribe_chunk_token_stream(encounter)

            while True:
                if store.get_encounter(encounter_id).live_cancel_requested:
                    break

                # LFM推論（ブロッキング）をスレッドプールで実行してイベントループを解放
                turn = await loop.run_in_executor(None, lambda: next(turn_iter, None))
                if turn is None:
                    break

                if store.get_encounter(encounter_id).live_cancel_requested:
                    break

                await asyncio.sleep(6.0)
                yield _sse("turn_start", {
                    "speaker": turn.speaker,
                    "speaker_label": SPEAKER_LABELS.get(turn.speaker, "不明"),
                    "start": turn.start,
                })
                stopped_during_turn = False
                for char in turn.text:
                    if store.get_encounter(encounter_id).live_cancel_requested:
                        stopped_during_turn = True
                        break
                    yield _sse("token", {"text": char})
                    await asyncio.sleep(0.01)  # タイピング演出
                if stopped_during_turn:
                    break
                yield _sse("turn_end", {"end": turn.end})

                latest = store.get_encounter(encounter_id)
                latest.transcript.append(turn)
                latest.transcript = normalize_transcript_turns(latest.transcript)
                latest.live_processed_chunks += 1
                latest.live_message = f"{latest.live_processed_chunks} turns processed"
                patient = store.get_patient(latest.patient_id)
                prompts = ClinicalTriggerEngine().run(ConversationBuffer().window(latest.transcript), patient)
                latest.clinical_prompts = merge_accumulated_prompts(latest.clinical_prompts, prompts)
                store.save_encounter(latest)

                for prompt in latest.clinical_prompts:
                    yield _sse("prompt", {
                        "kind": prompt.kind,
                        "kind_label": PROMPT_LABELS.get(prompt.kind, prompt.kind.upper()),
                        "title": prompt.title,
                        "detail": prompt.detail,
                        "priority": prompt.priority,
                    })

            done = store.get_encounter(encounter_id)
            if done.live_cancel_requested:
                done.live_status = "stopped"
                done.live_message = "文字起こしを停止しました。"
                done.live_cancel_requested = False
                store.save_encounter(done)
                yield _sse("status", {"stage": "stopped", "message": "文字起こしを停止しました。"})
                yield _sse("stopped", {"message": "文字起こしを停止しました。"})
            else:
                for audio in done.audio_sources:
                    audio.status = "transcribed"
                done.status = EncounterStatus.transcribed
                done.live_status = "done"
                done.live_message = "完了。カルテを作成してください。"
                store.save_encounter(done)
                yield _sse("status", {"stage": "done", "message": "完了。カルテを作成してください。"})
                yield _sse("done", {"message": "完了。カルテを作成してください。"})
        except Exception as exc:
            failed = store.get_encounter(encounter_id)
            failed.live_status = "error"
            failed.live_message = str(exc)
            store.save_encounter(failed)
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/encounters/{encounter_id}/soap")
async def generate_soap(encounter_id: str) -> RedirectResponse:
    encounter = store.get_encounter(encounter_id)
    patient = store.get_patient(encounter.patient_id)
    prompts = ClinicalTriggerEngine().run(ConversationBuffer().window(encounter.transcript), patient)
    encounter.clinical_prompts = merge_accumulated_prompts(encounter.clinical_prompts, prompts)
    encounter.chart = draft_soap(patient, encounter, encounter.transcript, encounter.clinical_prompts)
    encounter.status = EncounterStatus.draft
    store.save_encounter(encounter)
    return RedirectResponse(url=f"/encounters/{encounter.id}", status_code=303)


@router.get("/encounters/{encounter_id}/chart", response_class=HTMLResponse)
async def chart(encounter_id: str, request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chart.html", _context(encounter_id, request))


@router.get("/encounters/{encounter_id}/chart/edit", response_class=HTMLResponse)
async def chart_edit(encounter_id: str, request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chart_edit.html", _context(encounter_id, request))


@router.post("/encounters/{encounter_id}/chart/save")
async def chart_save(
    encounter_id: str,
    subjective: str = Form(""),
    objective: str = Form(""),
    assessment: str = Form(""),
    plan: str = Form(""),
    handoff: str = Form(""),
) -> RedirectResponse:
    encounter = store.get_encounter(encounter_id)
    encounter.chart = ChartDraft(subjective=subjective, objective=objective, assessment=assessment, plan=plan, handoff=handoff)
    encounter.status = EncounterStatus.draft
    store.save_encounter(encounter)
    return RedirectResponse(url=f"/encounters/{encounter.id}", status_code=303)


@router.post("/encounters/{encounter_id}/confirm")
async def confirm(encounter_id: str) -> RedirectResponse:
    encounter = store.set_status(encounter_id, EncounterStatus.confirmed)
    return RedirectResponse(url=f"/encounters/{encounter.id}", status_code=303)
