from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TriageCheck:
    name: str
    value: str
    flag: str  # "warn" | "ok" | "unknown"


@dataclass
class TriageResult:
    level: str   # "RED" | "YELLOW" | "GREEN" | "BLACK" | "未判定"
    label: str
    tone: str    # "red" | "yellow" | "green" | "black" | "pending"
    reason: str
    hits: list[str] = field(default_factory=list)
    checks: list[TriageCheck] = field(default_factory=list)


_RED_KW = [
    "呼吸困難", "呼吸なし", "無呼吸", "出血多量", "大量出血", "動脈出血",
    "意識消失", "意識不明", "ショック", "心停止", "重篤", "重傷",
    "血圧低下", "気道閉塞", "痙攣",
    "unconscious", "shock", "severe bleeding", "hemorrhage", "critical",
    "airway obstruction", "cardiac arrest",
]
_YELLOW_KW = [
    "骨折", "骨折疑い", "裂傷", "打撲", "痛み", "痛い", "吐き気", "嘔吐",
    "めまい", "頭痛", "腹痛", "外傷", "切傷", "受傷", "骨折",
    "pain", "laceration", "contusion", "nausea", "dizziness", "injury",
    "wound", "trauma", "fracture",
]
_GREEN_KW = [
    "歩行可能", "独歩", "自力歩行", "軽症", "軽傷",
    "ambulatory", "walking", "minor", "mild", "stable",
]
_BLACK_KW = [
    "死亡", "心肺停止", "心肺蘇生不能", "死",
    "dead", "deceased", "no pulse", "pulseless", "cpa",
]


def compute_triage(patient, encounter) -> TriageResult:
    chart = encounter.chart
    has_data = bool(
        chart.subjective or chart.objective or chart.assessment or chart.plan
        or encounter.transcript
    )
    if not has_data:
        return TriageResult(
            level="未判定", label="判定待ち", tone="pending",
            reason="文字起こし後にカルテを作成すると自動判定されます。",
        )

    text = " ".join([
        chart.subjective or "",
        chart.objective or "",
        chart.assessment or "",
        chart.plan or "",
        *(t.text for t in encounter.transcript),
    ]).lower()

    def hits(kws: list[str]) -> list[str]:
        return [kw for kw in kws if kw.lower() in text]

    black_h  = hits(_BLACK_KW)
    red_h    = hits(_RED_KW)
    yellow_h = hits(_YELLOW_KW)
    green_h  = hits(_GREEN_KW)
    checks   = _build_checks(text)

    if black_h:
        return TriageResult("BLACK", "黒タグ（死亡）", "black",
            "死亡または蘇生不可能を示すキーワードが検出されました。",
            black_h[:3], checks)
    if red_h:
        return TriageResult("RED", "高緊急度（即時）", "red",
            "生命を脅かすキーワードが検出されました。直ちに処置が必要です。",
            red_h[:3], checks)
    if yellow_h:
        return TriageResult("YELLOW", "準緊急（観察）", "yellow",
            "中程度の外傷または症状が確認されました。経過観察が必要です。",
            yellow_h[:3], checks)
    if green_h:
        return TriageResult("GREEN", "軽症（後回し）", "green",
            "歩行可能または軽症と判断されました。",
            green_h[:3], checks)
    return TriageResult("YELLOW", "要確認", "yellow",
        "情報は取得されましたが緊急度キーワードが不明確です。診察を継続してください。",
        [], checks)


def _build_checks(text: str) -> list[TriageCheck]:
    checks: list[TriageCheck] = []

    if any(k in text for k in ["呼吸なし", "無呼吸", "no breathing", "apnea"]):
        checks.append(TriageCheck("呼吸", "停止", "warn"))
    elif any(k in text for k in ["呼吸あり", "自発呼吸", "breathing", "breath"]):
        checks.append(TriageCheck("呼吸", "あり", "ok"))
    else:
        checks.append(TriageCheck("呼吸", "不明", "unknown"))

    if any(k in text for k in ["出血多量", "大量出血", "hemorrhage", "heavy bleeding"]):
        checks.append(TriageCheck("出血", "多量", "warn"))
    elif any(k in text for k in ["出血", "bleeding", "blood"]):
        checks.append(TriageCheck("出血", "あり", "warn"))
    else:
        checks.append(TriageCheck("出血", "不明", "unknown"))

    if any(k in text for k in ["意識なし", "意識消失", "意識不明", "unconscious"]):
        checks.append(TriageCheck("意識", "消失", "warn"))
    elif any(k in text for k in ["意識あり", "意識清明", "conscious", "alert", "awake"]):
        checks.append(TriageCheck("意識", "清明", "ok"))
    else:
        checks.append(TriageCheck("意識", "不明", "unknown"))

    if any(k in text for k in ["歩行可能", "独歩", "自力歩行", "walk", "ambulatory"]):
        checks.append(TriageCheck("歩行", "可能", "ok"))
    elif any(k in text for k in ["歩行不可", "歩けない", "cannot walk"]):
        checks.append(TriageCheck("歩行", "不可", "warn"))
    else:
        checks.append(TriageCheck("歩行", "不明", "unknown"))

    return checks
