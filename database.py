"""
Farmgate AI - Offline SQLite Database Manager
Handles local offline storage for grain assay batches, AGMARK quality results,
and Mandi trader feedback loop with Mean Absolute Error (MAE) tracking.
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
    Initializes the farmgate_history.db SQLite database.
    Creates table `batch_history` storing assay results and Mandi feedback.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                variety TEXT NOT NULL,
                total_grains INTEGER NOT NULL,
                whole_pct REAL NOT NULL,
                broken_pct REAL NOT NULL,
                discoloured_pct REAL NOT NULL,
                chalky_pct REAL NOT NULL,
                agmark_grade TEXT NOT NULL,
                mandi_broken_actual REAL DEFAULT NULL,
                mandi_price_offered REAL DEFAULT NULL,
                notes TEXT DEFAULT NULL
            )
            """
        )
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
    mandi_broken_actual: Optional[float] = None,
    mandi_price_offered: Optional[float] = None,
    notes: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: str = DB_NAME,
) -> int:
    """
    Saves a new grain test result into the SQLite database.
    Returns the generated batch ID.
    """
    init_db(db_path)
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO batch_history (
                timestamp, variety, total_grains, whole_pct, broken_pct,
                discoloured_pct, chalky_pct, agmark_grade,
                mandi_broken_actual, mandi_price_offered, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                variety,
                int(total_grains),
                float(round(whole_pct, 2)),
                float(round(broken_pct, 2)),
                float(round(discoloured_pct, 2)),
                float(round(chalky_pct, 2)),
                str(agmark_grade),
                float(round(mandi_broken_actual, 2)) if mandi_broken_actual is not None else None,
                float(round(mandi_price_offered, 2)) if mandi_price_offered is not None else None,
                notes,
            ),
        )
        conn.commit()
        batch_id = cursor.lastrowid
        return int(batch_id)
    finally:
        conn.close()


def update_mandi_feedback(
    batch_id: int,
    mandi_broken_actual: float,
    mandi_price_offered: float,
    notes: Optional[str] = None,
    db_path: str = DB_NAME,
) -> Dict[str, Any]:
    """
    Logs actual Mandi trader inspection results for a given batch.
    Calculates and returns updated Mean Absolute Error (MAE) across all logged feedback entries.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        if notes is not None:
            cursor.execute(
                """
                UPDATE batch_history
                SET mandi_broken_actual = ?, mandi_price_offered = ?, notes = ?
                WHERE id = ?
                """,
                (float(mandi_broken_actual), float(mandi_price_offered), notes, int(batch_id)),
            )
        else:
            cursor.execute(
                """
                UPDATE batch_history
                SET mandi_broken_actual = ?, mandi_price_offered = ?
                WHERE id = ?
                """,
                (float(mandi_broken_actual), float(mandi_price_offered), int(batch_id)),
            )
        conn.commit()
    finally:
        conn.close()

    mae_stats = calculate_feedback_mae(db_path)
    return mae_stats


def calculate_feedback_mae(db_path: str = DB_NAME) -> Dict[str, Any]:
    """
    Calculates the Mean Absolute Error (MAE) between app predicted broken_pct
    and trader actual mandi_broken_actual across all logged batches.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT broken_pct, mandi_broken_actual, mandi_price_offered
            FROM batch_history
            WHERE mandi_broken_actual IS NOT NULL
            """
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "mae": 0.0,
            "sample_count": 0,
            "mean_trader_price": 0.0,
            "details": [],
        }

    errors = []
    prices = []
    for r in rows:
        pred = r["broken_pct"]
        actual = r["mandi_broken_actual"]
        errors.append(abs(pred - actual))
        if r["mandi_price_offered"] is not None:
            prices.append(r["mandi_price_offered"])

    mae = float(sum(errors) / len(errors))
    mean_price = float(sum(prices) / len(prices)) if prices else 0.0

    return {
        "mae": round(mae, 2),
        "sample_count": len(rows),
        "mean_trader_price": round(mean_price, 2),
    }


def get_batch_history(db_path: str = DB_NAME) -> pd.DataFrame:
    """
    Fetches all past scans for side-by-side comparison and historical analysis.
    Returns a pandas DataFrame.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            id,
            timestamp,
            variety,
            total_grains,
            whole_pct,
            broken_pct,
            discoloured_pct,
            chalky_pct,
            agmark_grade,
            mandi_broken_actual,
            mandi_price_offered,
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
        cursor.execute(
            "SELECT * FROM batch_history WHERE id = ?",
            (int(batch_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def seed_sample_history_if_empty(db_path: str = DB_NAME) -> None:
    """Seeds initial demonstration data if the database is newly created."""
    init_db(db_path)
    df = get_batch_history(db_path)
    if len(df) == 0:
        seed_data = [
            (
                "Sona Masoori", 72, 88.9, 5.6, 2.8, 2.8, "Grade B (Standard / Good)",
                6.0, 3450.0, "Mandi lot #102 - Thanjavur Market", "2026-10-18 10:30:00"
            ),
            (
                "Basmati", 65, 92.3, 4.6, 1.5, 1.5, "Grade A (Special / Premium)",
                5.0, 5200.0, "Export procurement batch - Karnal", "2026-10-19 14:15:00"
            ),
            (
                "Swarna", 80, 71.2, 18.8, 5.0, 5.0, "Grade C (Fair Commercial)",
                20.0, 2600.0, "Direct farmer procurement - Madurai APMC", "2026-10-20 09:45:00"
            ),
            (
                "IR64", 75, 84.0, 10.7, 2.7, 2.7, "Grade B (Standard / Good)",
                11.5, 2900.0, "Kaveri delta harvest batch #12", "2026-10-21 16:20:00"
            ),
        ]
        for row in seed_data:
            save_batch(
                variety=row[0],
                total_grains=row[1],
                whole_pct=row[2],
                broken_pct=row[3],
                discoloured_pct=row[4],
                chalky_pct=row[5],
                agmark_grade=row[6],
                mandi_broken_actual=row[7],
                mandi_price_offered=row[8],
                notes=row[9],
                timestamp=row[10],
                db_path=db_path,
            )
