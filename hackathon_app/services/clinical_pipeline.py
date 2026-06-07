from __future__ import annotations

import re
from dataclasses import dataclass

from hackathon_app.models import ChartDraft, ClinicalPrompt, Encounter, Patient, TranscriptTurn


@dataclass
class PipelineResult:
    transcript: list[TranscriptTurn]
    prompts: list[ClinicalPrompt]
    chart: ChartDraft


class TranscriptBuffer:
    def __init__(self, pause_boundary_sec: float = 2.0, max_turn_sec: float = 30.0) -> None:
        self.pause_boundary_sec = pause_boundary_sec
        self.max_turn_sec = max_turn_sec
        self._turns: list[TranscriptTurn] = []

    def add_chunk(self, chunk: TranscriptTurn) -> None:
        text = _clean_text(chunk.text)
        if not text:
            return
        incoming = chunk.model_copy(update={"text": text})
        if not self._turns:
            self._turns.append(incoming)
            return

        current = self._turns[-1]
        if self._should_merge(current, incoming):
            current.text = f"{current.text} {incoming.text}".strip()
            current.end = incoming.end if incoming.end is not None else current.end
        else:
            self._turns.append(incoming)

    def finalize(self) -> list[TranscriptTurn]:
        return self._turns

    def _should_merge(self, left: TranscriptTurn, right: TranscriptTurn) -> bool:
        same_speaker = left.speaker == right.speaker or "unknown" in {left.speaker, right.speaker}
        if not same_speaker:
            return False
        if left.end is None or right.start is None or left.start is None:
            return left.speaker == "unknown" and right.speaker == "unknown"
        gap = right.start - left.end
        duration = (right.end or right.start) - left.start
        return gap <= self.pause_boundary_sec and duration <= self.max_turn_sec


class ConversationBuffer:
    def __init__(self, window_sec: float = 90.0) -> None:
        self.window_sec = window_sec

    def window(self, turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
        timed = [turn for turn in turns if turn.end is not None]
        if not timed:
            return turns[-8:]
        last_end = timed[-1].end or 0.0
        return [turn for turn in turns if turn.end is None or turn.end >= last_end - self.window_sec]


class ClinicalTriggerEngine:
    def run(self, turns: list[TranscriptTurn], patient: Patient) -> list[ClinicalPrompt]:
        text = _all_text(turns)
        prompts: list[ClinicalPrompt] = []

        prompts.extend(self._alerts(text))
        prompts.extend(self._negations(text))
        prompts.extend(self._need_checks(text, patient))
        prompts.extend(self._followups(text, patient))

        seen: set[tuple[str, str]] = set()
        unique: list[ClinicalPrompt] = []
        for prompt in sorted(prompts, key=lambda item: item.priority):
            key = (prompt.kind, prompt.title)
            if key in seen:
                continue
            seen.add(key)
            unique.append(prompt)
        return unique[:8]

    def _alerts(self, text: str) -> list[ClinicalPrompt]:
        rules = [
            (r"(38\.?[0-9]?|39\.?[0-9]?|発熱|高熱|fever)", "発熱", "体温、発症時刻、悪寒、解熱薬使用を確認", 1),
            (r"(胸痛|chest pain|息切れ|呼吸困難|shortness of breath)", "胸部症状", "胸痛・呼吸困難はバイタルと緊急度を優先確認", 1),
            (r"(片麻痺|ろれつ|意識障害|severe headache|weakness on one side)", "神経症状", "急性神経症状は発症時刻とFAST所見を確認", 1),
            (r"(出血|bleeding|blood)", "出血", "出血量、止血状況、抗凝固薬内服を確認", 2),
        ]
        return [ClinicalPrompt(kind="alert", title=title, detail=detail, priority=priority) for pat, title, detail, priority in rules if re.search(pat, text, re.I)]

    def _negations(self, text: str) -> list[ClinicalPrompt]:
        rules = [
            (r"(咳.*ない|咳嗽なし|no cough)", "咳嗽なし", "陰性所見としてSOAPに反映候補"),
            (r"(発熱.*ない|熱.*ない|no fever)", "発熱なし", "陰性所見としてSOAPに反映候補"),
            (r"(しびれ.*ない|しびれはありません|no numbness)", "しびれなし", "神経障害否定として記録候補"),
            (r"(アレルギー.*ない|no allergies|no known allergies)", "アレルギーなし", "薬剤・食物アレルギー欄に反映候補"),
        ]
        return [ClinicalPrompt(kind="negation", title=title, detail=detail, priority=2) for pat, title, detail in rules if re.search(pat, text, re.I)]

    def _need_checks(self, text: str, patient: Patient) -> list[ClinicalPrompt]:
        checks = []
        if not re.search(r"(アレルギー|allerg)", text, re.I):
            checks.append(ClinicalPrompt(kind="need_check", title="薬剤アレルギー", detail="薬剤・食物アレルギーの有無を確認", priority=3))
        if not re.search(r"(内服|薬|medication|medicine)", text, re.I):
            checks.append(ClinicalPrompt(kind="need_check", title="内服薬", detail="処方薬、市販薬、サプリを確認", priority=3))
        if not re.search(r"(いつ|何時|昨日|今日|発症|started|when|how long)", text, re.I):
            checks.append(ClinicalPrompt(kind="need_check", title="発症時刻", detail="症状や受傷の開始時刻を確認", priority=3))
        if patient.sex == "female" and patient.age >= 12 and patient.age <= 55 and not re.search(r"(妊娠|pregnan)", text, re.I):
            checks.append(ClinicalPrompt(kind="need_check", title="妊娠可能性", detail="検査・処方前に妊娠可能性を確認", priority=3))
        return checks

    def _followups(self, text: str, patient: Patient) -> list[ClinicalPrompt]:
        prompts = []
        if re.search(r"(喉|咽頭|sore throat|throat)", text, re.I):
            prompts.append(ClinicalPrompt(kind="follow_up", title="嚥下痛", detail="嚥下痛、開口障害、頸部腫脹を確認", priority=4))
        if re.search(r"(咳|cough|fever|発熱)", text, re.I):
            prompts.append(ClinicalPrompt(kind="follow_up", title="感染症状", detail="接触歴、鼻汁、咽頭痛、呼吸苦を確認", priority=4))
        if patient.category == "cut" or re.search(r"(切|創|包丁|cut|laceration)", text, re.I):
            prompts.append(ClinicalPrompt(kind="follow_up", title="創傷評価", detail="破傷風歴、異物混入、腱・神経血管障害を確認", priority=4))
        return prompts


def run_clinical_pipeline(patient: Patient, encounter: Encounter, raw_turns: list[TranscriptTurn]) -> PipelineResult:
    transcript_buffer = TranscriptBuffer()
    for turn in raw_turns:
        transcript_buffer.add_chunk(turn)
    transcript = transcript_buffer.finalize()

    conversation = ConversationBuffer().window(transcript)
    prompts = ClinicalTriggerEngine().run(conversation, patient)
    chart = draft_soap(patient, encounter, transcript, prompts)
    return PipelineResult(transcript=transcript, prompts=prompts, chart=chart)



def merge_accumulated_prompts(
    existing: list[ClinicalPrompt],
    incoming: list[ClinicalPrompt],
) -> list[ClinicalPrompt]:
    resolved_need_checks = _resolved_need_checks(incoming)
    merged: dict[tuple[str, str], ClinicalPrompt] = {}

    for prompt in existing:
        if prompt.kind == "need_check" and prompt.title in resolved_need_checks:
            continue
        merged[(prompt.kind, prompt.title)] = prompt

    for prompt in incoming:
        if prompt.kind == "need_check" and prompt.title in resolved_need_checks:
            continue
        key = (prompt.kind, prompt.title)
        current = merged.get(key)
        if current is None or prompt.priority < current.priority:
            merged[key] = prompt

    return sorted(merged.values(), key=lambda item: (item.priority, item.kind, item.title))[:12]


def _resolved_need_checks(prompts: list[ClinicalPrompt]) -> set[str]:
    titles = {prompt.title for prompt in prompts if prompt.kind == "negation"}
    resolved: set[str] = set()
    if "アレルギーなし" in titles:
        resolved.add("薬剤アレルギー")
    return resolved


def draft_soap(patient: Patient, encounter: Encounter, turns: list[TranscriptTurn], prompts: list[ClinicalPrompt]) -> ChartDraft:
    try:
        from hackathon_app.services.chart_lfm import generate_soap_with_lfm

        chart = generate_soap_with_lfm(patient, encounter, turns, prompts)
        if any([chart.subjective, chart.objective, chart.assessment, chart.plan]):
            return chart
    except Exception as exc:
        print(f"LFM chart generation fallback: {exc}")

    text = _all_text(turns)
    chief = patient.chief_complaint or _first_patient_like_text(turns) or "症状について相談あり。"

    negations = [p.title for p in prompts if p.kind == "negation"]
    alerts = [p.title for p in prompts if p.kind == "alert"]
    needs = [p.title for p in prompts if p.kind == "need_check"]

    subjective = chief
    if negations:
        subjective += " 否定所見: " + "、".join(negations) + "。"

    objective_parts = [image.finding for image in encounter.images if image.finding]
    if alerts:
        objective_parts.append("要注意所見: " + "、".join(alerts) + "。")
    objective = " ".join(objective_parts) or "バイタル、身体所見、画像所見を確認する。"

    assessment = _assessment_from_text(text, patient, alerts)
    plan = "追加問診と身体診察を行い、必要に応じて検査・処置を検討する。"
    if needs:
        plan += " 未確認項目: " + "、".join(needs) + "。"
    handoff = "Clinical Promptsを確認し、未確認項目を診察中に補完する。"
    return ChartDraft(subjective=subjective, objective=objective, assessment=assessment, plan=plan, handoff=handoff)


def _assessment_from_text(text: str, patient: Patient, alerts: list[str]) -> str:
    if patient.category == "cut" or re.search(r"(切|創|包丁|cut|laceration)", text, re.I):
        return "切創。腱損傷、神経血管損傷、感染リスクを鑑別。"
    if re.search(r"(咳|発熱|喉|sore throat|fever|cough)", text, re.I):
        return "急性上気道炎、咽頭炎、インフルエンザ/COVID-19などを鑑別。"
    if alerts:
        return "要注意所見を伴う症状。緊急度評価を優先。"
    return "問診内容に基づき鑑別を整理中。"


def _first_patient_like_text(turns: list[TranscriptTurn]) -> str:
    for turn in turns:
        if turn.speaker in {"patient", "unknown", "speaker_01"}:
            return turn.text
    return turns[0].text if turns else ""


def _all_text(turns: list[TranscriptTurn]) -> str:
    return " ".join(turn.text for turn in turns if turn.text)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
