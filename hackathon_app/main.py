from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hackathon_app.config import load_env_file
from hackathon_app.routes import encounters, patients
from hackathon_app.store import store


load_env_file()


def _prewarm() -> None:
    try:
        from hackathon_app.services.asr import _load_lfm
        _load_lfm()
        print("ASR prewarm done")
    except Exception as exc:
        print(f"ASR prewarm failed: {exc}")
    try:
        from hackathon_app.services.chart_lfm import _load_chart_model
        _load_chart_model()
        print("Chart prewarm done")
    except Exception as exc:
        print(f"Chart prewarm failed: {exc}")
    try:
        from hackathon_app.services.vision import _load_vlm
        _load_vlm()
        print("VLM prewarm done")
    except Exception as exc:
        print(f"VLM prewarm failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _prewarm)
    yield


app = FastAPI(title="Hackathon Clinical Copilot", lifespan=lifespan)
templates = Jinja2Templates(directory="hackathon_app/templates")

app.mount("/static", StaticFiles(directory="hackathon_app/static"), name="static")
app.mount("/images", StaticFiles(directory="hackathon_app/images"), name="images")
app.mount("/uploads", StaticFiles(directory="hackathon_app/storage/uploads"), name="uploads")

app.include_router(patients.router)
app.include_router(encounters.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Response:
    patients = store.list_patients()
    if patients:
        encounters = store.list_encounters_for_patient(patients[0].id)
        if encounters:
            return RedirectResponse(url=f"/encounters/{encounters[0].id}", status_code=303)
    return RedirectResponse(url="/start", status_code=303)


@app.get("/start", response_class=HTMLResponse)
async def start_page(request: Request) -> HTMLResponse:
    from hackathon_app.routes.encounters import _global_patient_context
    ctx = _global_patient_context()
    return templates.TemplateResponse("workspace.html", {
        "request": request,
        "patient": None,
        "encounter": None,
        "triage": None,
        "speaker_labels": {},
        "encounter_number": 0,
        "auto_open_reg": True,
        **ctx,
    })
