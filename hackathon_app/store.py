from __future__ import annotations

import json
import shutil
from pathlib import Path
from threading import RLock

from hackathon_app.models import Encounter, EncounterStatus, Patient


class HackathonStore:
    def __init__(self, path: str = "hackathon_app/storage/db.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_root = self.path.parent / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        if not self.path.exists():
            self._write({"patients": [], "encounters": []})

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"patients": [], "encounters": []}

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def upload_dir(self, encounter_id: str) -> Path:
        path = self.upload_root / encounter_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_patients(self) -> list[Patient]:
        with self._lock:
            return [Patient.model_validate(item) for item in self._read().get("patients", [])]

    def list_encounters(self) -> list[Encounter]:
        with self._lock:
            return [Encounter.model_validate(item) for item in self._read().get("encounters", [])]

    def raw_encounter(self, encounter_id: str) -> dict:
        for item in self._read().get("encounters", []):
            if item.get("id") == encounter_id:
                return item
        raise KeyError(encounter_id)

    def _legacy_data(self) -> dict:
        return {"patients": [], "encounters": []}

    def _import_legacy_patient(self, patient_id: str) -> None:
        legacy = self._legacy_data()
        patient_raw = next((item for item in legacy.get("patients", []) if item.get("id") == patient_id), None)
        if patient_raw is None:
            return
        with self._lock:
            data = self._read()
            if any(item.get("id") == patient_id for item in data.get("patients", [])):
                return
            data.setdefault("patients", []).append(patient_raw)
            self._write(data)

    def _import_legacy_encounter(self, encounter_id: str) -> None:
        legacy = self._legacy_data()
        encounter_raw = next((item for item in legacy.get("encounters", []) if item.get("id") == encounter_id), None)
        if encounter_raw is None:
            return
        patient_id = encounter_raw.get("patient_id")
        if patient_id:
            self._import_legacy_patient(patient_id)
        with self._lock:
            data = self._read()
            if any(item.get("id") == encounter_id for item in data.get("encounters", [])):
                return
            patient_encounters = [item for item in data.get("encounters", []) if item.get("patient_id") == patient_id]
            imported = dict(encounter_raw)
            imported.setdefault("encounter_number", max([int(item.get("encounter_number", 1)) for item in patient_encounters] or [0]) + 1)
            data.setdefault("encounters", []).append(imported)
            self._write(data)
            self.upload_dir(encounter_id)

    def encounter_number(self, encounter_id: str) -> int:
        return int(self.raw_encounter(encounter_id).get("encounter_number", 1))

    def get_patient(self, patient_id: str) -> Patient:
        for patient in self.list_patients():
            if patient.id == patient_id:
                return patient
        self._import_legacy_patient(patient_id)
        for patient in self.list_patients():
            if patient.id == patient_id:
                return patient
        raise KeyError(patient_id)

    def get_encounter(self, encounter_id: str) -> Encounter:
        for encounter in self.list_encounters():
            if encounter.id == encounter_id:
                return encounter
        self._import_legacy_encounter(encounter_id)
        for encounter in self.list_encounters():
            if encounter.id == encounter_id:
                return encounter
        raise KeyError(encounter_id)

    def list_encounters_for_patient(self, patient_id: str) -> list[Encounter]:
        encounters = [enc for enc in self.list_encounters() if enc.patient_id == patient_id]
        return sorted(encounters, key=lambda enc: (self.encounter_number(enc.id), enc.created_at), reverse=True)

    def create_patient_with_encounter(self, patient: Patient) -> Encounter:
        with self._lock:
            data = self._read()
            encounter = Encounter(patient_id=patient.id)
            encounter_raw = encounter.model_dump(mode="json")
            encounter_raw["encounter_number"] = 1
            data.setdefault("patients", []).append(patient.model_dump(mode="json"))
            data.setdefault("encounters", []).append(encounter_raw)
            self._write(data)
            self.upload_dir(encounter.id)
            return encounter

    def create_encounter_for_patient(self, patient_id: str, number: int | None = None) -> Encounter:
        with self._lock:
            data = self._read()
            existing = [item for item in data.get("encounters", []) if item.get("patient_id") == patient_id]
            next_number = number or (max([int(item.get("encounter_number", 1)) for item in existing] or [0]) + 1)
            encounter = Encounter(patient_id=patient_id)
            encounter_raw = encounter.model_dump(mode="json")
            encounter_raw["encounter_number"] = next_number
            data.setdefault("encounters", []).append(encounter_raw)
            self._write(data)
            self.upload_dir(encounter.id)
            return encounter

    def save_patient(self, patient: Patient) -> Patient:
        with self._lock:
            data = self._read()
            for idx, item in enumerate(data.get("patients", [])):
                if item.get("id") == patient.id:
                    data["patients"][idx] = patient.model_dump(mode="json")
                    self._write(data)
                    return patient
        raise KeyError(patient.id)

    def save_encounter(self, encounter: Encounter) -> Encounter:
        with self._lock:
            data = self._read()
            for idx, item in enumerate(data.get("encounters", [])):
                if item.get("id") == encounter.id:
                    updated = encounter.model_dump(mode="json")
                    updated["encounter_number"] = item.get("encounter_number", 1)
                    data["encounters"][idx] = updated
                    self._write(data)
                    return encounter
        raise KeyError(encounter.id)

    def delete_patient(self, patient_id: str) -> None:
        with self._lock:
            data = self._read()
            encounter_ids = [item.get("id") for item in data.get("encounters", []) if item.get("patient_id") == patient_id]
            data["patients"] = [item for item in data.get("patients", []) if item.get("id") != patient_id]
            data["encounters"] = [item for item in data.get("encounters", []) if item.get("patient_id") != patient_id]
            self._write(data)
        for encounter_id in encounter_ids:
            if encounter_id:
                shutil.rmtree(self.upload_root / encounter_id, ignore_errors=True)

    def set_status(self, encounter_id: str, status: EncounterStatus) -> Encounter:
        encounter = self.get_encounter(encounter_id)
        encounter.status = status
        return self.save_encounter(encounter)


store = HackathonStore()
