"""
Farmgate AI - Offline SQLite Database Manager (Feature 6: Mandi Feedback Loop)
Handles local offline storage for grain assay batches, AGMARK quality results,
and Mandi trader / hand-counted feedback loop with Mean Absolute Error (MAE) tracking.

Schema:
Table `mandi_feedback`:
- sample_id: TEXT (Primary Key or unique identifier, e.g. SMP-1001)
- timestamp: TEXT (YYYY-MM-DD HH:MM:SS)
- variety: TEXT (e.g. Basmati, Sona Masoori, Swarna, IR64)
- app_broken_pct: REAL (App-estimated broken grain percentage)
- actual_broken_pct: REAL (Ground-truth actual broken % entered by user or trader)
- capture_quality_status: TEXT (Passed / Warning / Failed)
- notes: TEXT (Lighting conditions, angle, trader notes)

Table `batch_history`:
Maintains full morphometric scan metrics for history comparison.
"""

import sqlite3
import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


DB_NAME = "farmgate_history.db"


def get_connection(db_path: str = DB_NAME) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_NAME) -> None:
    """
    Initializes the SQLite database with `batch_history` and `mandi_feedback` tables.
    Runs automatic schema migrations for existing databases.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # 1. Main batch history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT DEFAULT NULL,
                timestamp TEXT NOT NULL,
                variety TEXT NOT NULL,
                total_grains INTEGER NOT NULL,
                whole_pct REAL NOT NULL,
                broken_pct REAL NOT NULL,
                discoloured_pct REAL NOT NULL,
                chalky_pct REAL NOT NULL,
                agmark_grade TEXT NOT NULL,
                actual_broken_pct REAL DEFAULT NULL,
                mandi_broken_actual REAL DEFAULT NULL,
                capture_quality_status TEXT DEFAULT 'Passed',
                notes TEXT DEFAULT NULL
            )
            """
        )

        # 2. Feature 6 Mandi Feedback Table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mandi_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                variety TEXT NOT NULL,
                app_broken_pct REAL NOT NULL,
                actual_broken_pct REAL DEFAULT NULL,
                capture_quality_status TEXT DEFAULT 'Passed',
                notes TEXT DEFAULT NULL
            )
            """
        )

        # Migration: Ensure columns exist in batch_history if created from older schema
        cursor.execute("PRAGMA table_info(batch_history)")
        columns = [row["name"] for row in cursor.fetchall()]

        if "sample_id" not in columns:
            cursor.execute("ALTER TABLE batch_history ADD COLUMN sample_id TEXT DEFAULT NULL")
        if "actual_broken_pct" not in columns:
            cursor.execute("ALTER TABLE batch_history ADD COLUMN actual_broken_pct REAL DEFAULT NULL")
        if "capture_quality_status" not in columns:
            cursor.execute("ALTER TABLE batch_history ADD COLUMN capture_quality_status TEXT DEFAULT 'Passed'")

        conn.commit()
    finally:
        conn.close()


def save_batch(
    variety: str,
    total_grains: int,
    whole_pct: float,
    broken_pct: float,
    discoloured_pct: float,
    chalky_pct: float,
    agmark_grade: str,
    sample_id: Optional[str] = None,
    capture_quality_status: str = "Passed",
    actual_broken_pct: Optional[float] = None,
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
    mandi_broken_actual: Optional[float] = None,
    mandi_price_offered: Optional[float] = None,
    db_path: str = DB_NAME,
) -> int:
    """
    Saves a newly assayed batch to SQLite.
    Synchronizes into both `batch_history` and `mandi_feedback` tables.
    Returns the integer batch ID.
    """
    init_db(db_path)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    actual_val = actual_broken_pct if actual_broken_pct is not None else mandi_broken_actual

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO batch_history (
                sample_id, timestamp, variety, total_grains, whole_pct, broken_pct,
                discoloured_pct, chalky_pct, agmark_grade,
                actual_broken_pct, mandi_broken_actual, capture_quality_status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                timestamp,
                variety,
                int(total_grains),
                float(round(whole_pct, 2)),
                float(round(broken_pct, 2)),
                float(round(discoloured_pct, 2)),
                float(round(chalky_pct, 2)),
                str(agmark_grade),
                float(round(actual_val, 2)) if actual_val is not None else None,
                float(round(actual_val, 2)) if actual_val is not None else None,
                capture_quality_status,
                notes,
            ),
        )
        batch_id = cursor.lastrowid

        # Determine effective sample_id
        eff_sample_id = sample_id if sample_id else f"SMP-{batch_id:04d}"
        if sample_id is None:
            cursor.execute("UPDATE batch_history SET sample_id = ? WHERE id = ?", (eff_sample_id, batch_id))

        # Insert or replace into mandi_feedback
        cursor.execute(
            """
            INSERT INTO mandi_feedback (
                sample_id, timestamp, variety, app_broken_pct,
                actual_broken_pct, capture_quality_status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sample_id) DO UPDATE SET
                app_broken_pct = excluded.app_broken_pct,
                capture_quality_status = excluded.capture_quality_status,
                notes = COALESCE(excluded.notes, mandi_feedback.notes)
            """,
            (
                eff_sample_id,
                timestamp,
                variety,
                float(round(broken_pct, 2)),
                float(round(actual_val, 2)) if actual_val is not None else None,
                capture_quality_status,
                notes,
            ),
        )
        conn.commit()
        return int(batch_id)
    finally:
        conn.close()


def record_feedback(
    sample_id: str,
    actual_broken_pct: float,
    notes: Optional[str] = None,
    db_path: str = DB_NAME,
) -> Dict[str, Any]:
    """
    Feature 6: Records or updates trader / hand-counted actual broken % for a sample.
    Calculates and returns updated MAE statistics.
    """
    init_db(db_path)
    actual_val = float(round(actual_broken_pct, 2))
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        # Update mandi_feedback table
        cursor.execute(
            """
            UPDATE mandi_feedback
            SET actual_broken_pct = ?,
                notes = CASE WHEN ? IS NOT NULL THEN ? ELSE notes END
            WHERE sample_id = ?
            """,
            (actual_val, notes, notes, sample_id),
        )

        # Also update batch_history table
        cursor.execute(
            """
            UPDATE batch_history
            SET actual_broken_pct = ?,
                mandi_broken_actual = ?,
                notes = CASE WHEN ? IS NOT NULL THEN ? ELSE notes END
            WHERE sample_id = ? OR id = ?
            """,
            (
                actual_val,
                actual_val,
                notes,
                notes,
                sample_id,
                int(sample_id.replace("SMP-", "")) if sample_id.startswith("SMP-") and sample_id.replace("SMP-", "").isdigit() else -1,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return calculate_feedback_mae(db_path)


def update_mandi_feedback(
    batch_id: int,
    mandi_broken_actual: float,
    mandi_price_offered: Optional[float] = None,
    notes: Optional[str] = None,
    db_path: str = DB_NAME,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for updating feedback by integer batch ID."""
    sample_id = f"SMP-{batch_id:04d}"
    return record_feedback(
        sample_id=sample_id,
        actual_broken_pct=mandi_broken_actual,
        notes=notes,
        db_path=db_path,
    )


def calculate_feedback_mae(db_path: str = DB_NAME) -> Dict[str, Any]:
    """
    Feature 6: Computes Mean Absolute Error (MAE) across all saved samples
    that have an entered actual broken percentage:
    MAE = mean(|app_broken_pct - actual_broken_pct|)
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sample_id, app_broken_pct, actual_broken_pct
            FROM mandi_feedback
            WHERE actual_broken_pct IS NOT NULL
            """
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "mae": 0.0,
            "sample_count": 0,
            "details": [],
        }

    errors = []
    details = []
    for r in rows:
        err = abs(float(r["app_broken_pct"]) - float(r["actual_broken_pct"]))
        errors.append(err)
        details.append({
            "sample_id": r["sample_id"],
            "predicted": r["app_broken_pct"],
            "actual": r["actual_broken_pct"],
            "error": round(err, 2),
        })

    mae = float(sum(errors) / len(errors))
    return {
        "mae": round(mae, 2),
        "sample_count": len(rows),
        "details": details,
    }


def get_feedback_comparison_df(db_path: str = DB_NAME) -> pd.DataFrame:
    """
    Feature 6: Returns a structured comparison DataFrame of all samples showing:
    sample_id, timestamp, variety, app_broken_pct, actual_broken_pct, absolute_error,
    capture_quality_status, and notes.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            sample_id,
            timestamp,
            variety,
            app_broken_pct,
            actual_broken_pct,
            capture_quality_status,
            notes
        FROM mandi_feedback
        ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=[
            "sample_id", "timestamp", "variety",
            "app_broken_pct", "actual_broken_pct", "absolute_error",
            "capture_quality_status", "notes"
        ])

    # Compute Absolute Error column
    df["absolute_error"] = df.apply(
        lambda row: round(abs(row["app_broken_pct"] - row["actual_broken_pct"]), 2)
        if pd.notnull(row["actual_broken_pct"]) else None,
        axis=1
    )
    return df


def export_feedback_csv(output_csv_path: str = "mandi_feedback_log.csv", db_path: str = DB_NAME) -> str:
    """
    Feature 6: Exports the full Mandi feedback comparison table to CSV.
    """
    df = get_feedback_comparison_df(db_path)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def get_batch_history(db_path: str = DB_NAME) -> pd.DataFrame:
    """Fetches all past scans for side-by-side comparison."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            id,
            sample_id,
            timestamp,
            variety,
            total_grains,
            whole_pct,
            broken_pct,
            discoloured_pct,
            chalky_pct,
            agmark_grade,
            actual_broken_pct,
            capture_quality_status,
            notes
        FROM batch_history
        ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


def get_batch_by_id(batch_id: int, db_path: str = DB_NAME) -> Optional[Dict[str, Any]]:
    """Returns a single batch dictionary by its ID."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_history WHERE id = ?", (int(batch_id),))
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def seed_sample_history_if_empty(db_path: str = DB_NAME) -> None:
    """Seeds initial demonstration data if database is newly created (strictly no price numbers)."""
    init_db(db_path)
    df = get_feedback_comparison_df(db_path)
    if len(df) == 0:
        seed_data = [
            (
                "SMP-1001", "Sona Masoori", 72, 88.9, 5.6, 2.8, 2.8, "Grade B (Standard / Good)",
                6.0, "Passed", "Diffuse natural morning lighting, 90 deg perpendicular angle", "2026-10-18 10:30:00"
            ),
            (
                "SMP-1002", "Basmati", 65, 92.3, 4.6, 1.5, 1.5, "Grade A (Special / Premium)",
                5.0, "Passed", "Even overhead white illumination, perpendicular", "2026-10-19 14:15:00"
            ),
            (
                "SMP-1003", "Swarna", 80, 71.2, 18.8, 5.0, 5.0, "Grade C (Fair Commercial)",
                20.0, "Warning", "Minor shadow across bottom-right quadrant, slight 15 deg tilt", "2026-10-20 09:45:00"
            ),
            (
                "SMP-1004", "IR64", 75, 84.0, 10.7, 2.7, 2.7, "Grade B (Standard / Good)",
                11.5, "Passed", "Overhead LED desk lamp, flat orientation", "2026-10-21 16:20:00"
            ),
        ]
        for row in seed_data:
            save_batch(
                sample_id=row[0],
                variety=row[1],
                total_grains=row[2],
                whole_pct=row[3],
                broken_pct=row[4],
                discoloured_pct=row[5],
                chalky_pct=row[6],
                agmark_grade=row[7],
                actual_broken_pct=row[8],
                capture_quality_status=row[9],
                notes=row[10],
                timestamp=row[11],
                db_path=db_path,
            )
