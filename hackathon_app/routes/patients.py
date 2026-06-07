from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from hackathon_app.models import Patient, PatientCategory
from hackathon_app.store import store

router = APIRouter()
templates = Jinja2Templates(directory="hackathon_app/templates")

STATUS_LABELS = {
    "intake": "受付",
    "recording": "文字起こし中",
    "transcribed": "転写済み",
    "draft": "カルテ作成済",
    "confirmed": "確認済み",
}


@router.post("/patients")
async def create_patient(
    name: str = Form(...),
    age: int = Form(...),
    sex: str = Form("unknown"),
    category: str = Form("other"),
    chief_complaint: str = Form(""),
    memo: str = Form(""),
) -> RedirectResponse:
    patient = Patient(
        name=name,
        age=age,
        sex=sex,
        category=PatientCategory(category),
        chief_complaint=chief_complaint,
        memo=memo,
    )
    encounter = store.create_patient_with_encounter(patient)
    return RedirectResponse(url=f"/encounters/{encounter.id}", status_code=303)


@router.post("/patients/{patient_id}/encounters")
async def create_followup(patient_id: str) -> RedirectResponse:
    encounter = store.create_encounter_for_patient(patient_id)
    return RedirectResponse(url=f"/encounters/{encounter.id}", status_code=303)


@router.post("/patients/{patient_id}/delete")
async def delete_patient(patient_id: str) -> RedirectResponse:
    store.delete_patient(patient_id)
    patients = store.list_patients()
    if patients:
        enc = store.list_encounters_for_patient(patients[0].id)
        if enc:
            return RedirectResponse(url=f"/encounters/{enc[0].id}", status_code=303)
    return RedirectResponse(url="/", status_code=303)
