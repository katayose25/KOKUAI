from __future__ import annotations

from hackathon_app.models import ChartDraft, Encounter, ImageAsset, Patient, TranscriptTurn


def describe_image(image: ImageAsset) -> str:
    name = image.filename.lower()
    if "burn" in name or "yakedo" in name:
        return "患部に発赤を認める。水疱の有無は確認が必要。"
    if "bruise" in name or "打撲" in name:
        return "皮下出血を疑う色調変化を認める。腫脹の程度は診察で確認。"
    return "創部の画像が添付されている。出血、腫脹、感染兆候の確認が必要。"


def transcribe(_: Encounter) -> list[TranscriptTurn]:
    return [
        TranscriptTurn(speaker="doctor", start=0.4, end=3.2, text="今日はどうされましたか。"),
        TranscriptTurn(speaker="patient", start=3.8, end=9.6, text="包丁で指を切ってしまいました。"),
        TranscriptTurn(speaker="doctor", start=10.1, end=13.4, text="いつ頃のけがですか。"),
        TranscriptTurn(speaker="patient", start=13.8, end=18.2, text="三十分くらい前です。まだ少し出血しています。"),
        TranscriptTurn(speaker="doctor", start=18.8, end=24.0, text="しびれや指の動かしにくさはありますか。"),
        TranscriptTurn(speaker="patient", start=24.4, end=28.0, text="しびれはありません。指は動かせます。"),
    ]


def create_chart(patient: Patient, encounter: Encounter) -> ChartDraft:
    findings = " ".join(image.finding for image in encounter.images if image.finding)
    transcript = " ".join(turn.text for turn in encounter.transcript)
    subjective = patient.chief_complaint or "患者より外傷に関する訴えあり。"
    if "切" in transcript:
        subjective = "包丁で指を切ったとの訴え。受傷は約30分前。しびれは否定。"
    objective = findings or "創部の状態、出血量、腫脹、神経血管障害の有無を確認する。"
    assessment = "指切創。腱損傷、神経血管損傷、感染リスクの評価が必要。"
    plan = "創部洗浄、止血、必要に応じて縫合を検討。破傷風ワクチン歴を確認し、感染徴候と再診目安を説明する。"
    return ChartDraft(
        subjective=subjective,
        objective=objective,
        assessment=assessment,
        plan=plan,
        handoff="疼痛、出血持続、しびれ、発赤や腫脹増悪があれば再評価。",
    )
