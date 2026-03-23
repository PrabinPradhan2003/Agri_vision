"""SQLite-backed scan history for the Flask app."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  image_filename TEXT NOT NULL,
  pred_idx INTEGER,
  pred_label TEXT,
  confidence REAL,
  top_predictions_json TEXT,
    weather_json TEXT,
    severity_percent REAL,
    severity_level TEXT,
    model_name TEXT,
    model_weights TEXT,
        location_text TEXT,
    feedback_correct INTEGER,
    feedback_label TEXT
);
"""


@dataclass(frozen=True)
class ScanRow:
    id: int
    created_at: str
    image_filename: str
    pred_idx: Optional[int]
    pred_label: Optional[str]
    confidence: Optional[float]
    top_predictions: List[Dict[str, Any]]
    weather: Optional[Dict[str, Any]]
    severity_percent: Optional[float]
    severity_level: Optional[str]
    model_name: Optional[str]
    model_weights: Optional[str]
    location_text: Optional[str]
    feedback_correct: Optional[bool]
    feedback_label: Optional[str]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(DB_SCHEMA)
        # Lightweight migrations for older DBs.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(scans)").fetchall()}

        def _add_col(name: str, ddl_type: str) -> None:
            if name in cols:
                return
            conn.execute(f"ALTER TABLE scans ADD COLUMN {name} {ddl_type}")
            cols.add(name)

        _add_col("severity_percent", "REAL")
        _add_col("severity_level", "TEXT")
        _add_col("model_name", "TEXT")
        _add_col("model_weights", "TEXT")
        _add_col("location_text", "TEXT")
        _add_col("feedback_correct", "INTEGER")
        _add_col("feedback_label", "TEXT")
        conn.commit()


def add_scan(
    db_path: Path,
    *,
    image_filename: str,
    pred_idx: Optional[int],
    pred_label: Optional[str],
    confidence: Optional[float],
    top_predictions: List[Dict[str, Any]],
    weather: Optional[Dict[str, Any]],
    severity_percent: Optional[float] = None,
    severity_level: Optional[str] = None,
    model_name: Optional[str] = None,
    model_weights: Optional[str] = None,
    location_text: Optional[str] = None,
) -> None:
    init_db(db_path)
    created_at = datetime.utcnow().isoformat() + "Z"

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scans (
              created_at, image_filename, pred_idx, pred_label, confidence,
              top_predictions_json, weather_json,
              severity_percent, severity_level,
                            model_name, model_weights,
                            location_text
            )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                image_filename,
                pred_idx,
                pred_label,
                confidence,
                json.dumps(top_predictions),
                json.dumps(weather) if weather is not None else None,
                severity_percent,
                severity_level,
                model_name,
                model_weights,
                                location_text,
            ),
        )
        conn.commit()


def add_scan_return_id(
    db_path: Path,
    *,
    image_filename: str,
    pred_idx: Optional[int],
    pred_label: Optional[str],
    confidence: Optional[float],
    top_predictions: List[Dict[str, Any]],
    weather: Optional[Dict[str, Any]],
    severity_percent: Optional[float] = None,
    severity_level: Optional[str] = None,
    model_name: Optional[str] = None,
    model_weights: Optional[str] = None,
    location_text: Optional[str] = None,
) -> int:
    """Insert a scan and return its row id."""
    init_db(db_path)
    created_at = datetime.utcnow().isoformat() + "Z"
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scans (
              created_at, image_filename, pred_idx, pred_label, confidence,
              top_predictions_json, weather_json,
              severity_percent, severity_level,
                            model_name, model_weights,
                            location_text
            )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                image_filename,
                pred_idx,
                pred_label,
                confidence,
                json.dumps(top_predictions),
                json.dumps(weather) if weather is not None else None,
                severity_percent,
                severity_level,
                model_name,
                model_weights,
                                location_text,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_scan(db_path: Path, scan_id: int) -> Optional[ScanRow]:
    init_db(db_path)
    with _connect(db_path) as conn:
        r = conn.execute("SELECT * FROM scans WHERE id = ?", (int(scan_id),)).fetchone()
        if r is None:
            return None

    top_json = r["top_predictions_json"]
    weather_json = r["weather_json"]
    return ScanRow(
        id=int(r["id"]),
        created_at=str(r["created_at"]),
        image_filename=str(r["image_filename"]),
        pred_idx=(int(r["pred_idx"]) if r["pred_idx"] is not None else None),
        pred_label=(str(r["pred_label"]) if r["pred_label"] is not None else None),
        confidence=(float(r["confidence"]) if r["confidence"] is not None else None),
        top_predictions=(json.loads(top_json) if top_json else []),
        weather=(json.loads(weather_json) if weather_json else None),
        severity_percent=(float(r["severity_percent"]) if r["severity_percent"] is not None else None),
        severity_level=(str(r["severity_level"]) if r["severity_level"] is not None else None),
        model_name=(str(r["model_name"]) if r["model_name"] is not None else None),
        model_weights=(str(r["model_weights"]) if r["model_weights"] is not None else None),
        location_text=(str(r["location_text"]) if r["location_text"] is not None else None),
        feedback_correct=(bool(int(r["feedback_correct"])) if r["feedback_correct"] is not None else None),
        feedback_label=(str(r["feedback_label"]) if r["feedback_label"] is not None else None),
    )


def set_feedback(
    db_path: Path,
    *,
    scan_id: int,
    correct: bool,
    correct_label: Optional[str] = None,
) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE scans SET feedback_correct = ?, feedback_label = ? WHERE id = ?",
            (1 if correct else 0, correct_label, int(scan_id)),
        )
        conn.commit()


def list_scans(db_path: Path, limit: int = 25) -> List[ScanRow]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()

    out: List[ScanRow] = []
    for r in rows:
        top_json = r["top_predictions_json"]
        weather_json = r["weather_json"]
        out.append(
            ScanRow(
                id=int(r["id"]),
                created_at=str(r["created_at"]),
                image_filename=str(r["image_filename"]),
                pred_idx=(int(r["pred_idx"]) if r["pred_idx"] is not None else None),
                pred_label=(str(r["pred_label"]) if r["pred_label"] is not None else None),
                confidence=(float(r["confidence"]) if r["confidence"] is not None else None),
                top_predictions=(json.loads(top_json) if top_json else []),
                weather=(json.loads(weather_json) if weather_json else None),
                severity_percent=(float(r["severity_percent"]) if r["severity_percent"] is not None else None),
                severity_level=(str(r["severity_level"]) if r["severity_level"] is not None else None),
                model_name=(str(r["model_name"]) if r["model_name"] is not None else None),
                model_weights=(str(r["model_weights"]) if r["model_weights"] is not None else None),
                location_text=(str(r["location_text"]) if r["location_text"] is not None else None),
                feedback_correct=(bool(int(r["feedback_correct"])) if r["feedback_correct"] is not None else None),
                feedback_label=(str(r["feedback_label"]) if r["feedback_label"] is not None else None),
            )
        )

    return out


def delete_scan(db_path: Path, scan_id: int) -> bool:
    """Delete a scan by id. Returns True if a row was deleted."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM scans WHERE id = ?", (int(scan_id),))
        conn.commit()
        return int(cur.rowcount or 0) > 0
