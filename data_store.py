import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd


class DataVistaStore:
    """SQLite-backed persistence for shared application data."""

    def __init__(self, db_path: Path, students_json_path: Path, gdp_csv_path: Path):
        self.db_path = Path(db_path)
        self.students_json_path = Path(students_json_path)
        self.gdp_csv_path = Path(gdp_csv_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    name TEXT PRIMARY KEY,
                    grade REAL NOT NULL,
                    range_upper REAL NOT NULL DEFAULT 10,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gdp_data (
                    country_code TEXT NOT NULL,
                    country_name TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    gdp REAL,
                    PRIMARY KEY (country_code, year)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gdp_year ON gdp_data(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gdp_country_name ON gdp_data(country_name)")
            conn.commit()

        self._ensure_student_range_upper_column()

        self._migrate_students_json_if_needed()
        self._seed_gdp_if_needed()

    def _ensure_student_range_upper_column(self) -> None:
        with self._connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(students)").fetchall()}
            if "range_upper" in columns:
                return

            conn.execute("ALTER TABLE students ADD COLUMN range_upper REAL NOT NULL DEFAULT 10")
            conn.execute(
                """
                UPDATE students
                SET range_upper = CASE
                    WHEN grade > 10 THEN 100.0
                    ELSE 10.0
                END
                """
            )
            conn.commit()

    @staticmethod
    def _default_range_upper(grade: float) -> float:
        return 100.0 if float(grade) > 10 else 10.0

    def _migrate_students_json_if_needed(self) -> None:
        if not self.students_json_path.exists():
            return

        with self._connect() as conn:
            current_count = conn.execute("SELECT COUNT(*) AS count FROM students").fetchone()["count"]
            if current_count > 0:
                return

        try:
            payload = json.loads(self.students_json_path.read_text(encoding="utf-8"))
        except Exception:
            return

        if not isinstance(payload, dict) or not payload:
            return

        records = []
        for name, grade in payload.items():
            key = str(name).strip()
            if not key:
                continue
            try:
                numeric_grade = float(grade)
            except (TypeError, ValueError):
                continue
            records.append((key, numeric_grade, self._default_range_upper(numeric_grade)))

        if not records:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO students (name, grade, range_upper)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    grade = excluded.grade,
                    range_upper = excluded.range_upper,
                    updated_at = CURRENT_TIMESTAMP
                """,
                records,
            )
            conn.commit()

    def _seed_gdp_if_needed(self) -> None:
        with self._connect() as conn:
            existing_rows = conn.execute("SELECT COUNT(*) AS count FROM gdp_data").fetchone()["count"]
            if existing_rows > 0:
                return

        if not self.gdp_csv_path.exists():
            return

        raw_df = pd.read_csv(self.gdp_csv_path)
        required = {"Country Code", "Country Name"}
        if not required.issubset(raw_df.columns):
            return

        year_columns = [col for col in raw_df.columns if str(col).isdigit()]
        if not year_columns:
            return

        gdp_df = raw_df.melt(
            ["Country Code", "Country Name"],
            year_columns,
            "Year",
            "GDP",
        )
        gdp_df["Year"] = pd.to_numeric(gdp_df["Year"], errors="coerce")
        gdp_df["GDP"] = pd.to_numeric(gdp_df["GDP"], errors="coerce")
        gdp_df = gdp_df.dropna(subset=["Year", "GDP"])

        records = [
            (
                str(row["Country Code"]),
                str(row["Country Name"]),
                int(row["Year"]),
                float(row["GDP"]),
            )
            for _, row in gdp_df.iterrows()
        ]

        if not records:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO gdp_data (country_code, country_name, year, gdp)
                VALUES (?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def list_students(self) -> List[Dict[str, float]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name, grade, range_upper FROM students ORDER BY LOWER(name)").fetchall()
        return [
            {
                "name": row["name"],
                "grade": float(row["grade"]),
                "range_upper": float(row["range_upper"] if row["range_upper"] is not None else self._default_range_upper(row["grade"])),
            }
            for row in rows
        ]

    def upsert_student(self, name: str, grade: float, range_upper: float = 10.0) -> None:
        key = name.strip()
        if not key:
            raise ValueError("Student name cannot be empty.")

        try:
            range_value = float(range_upper)
        except (TypeError, ValueError) as exc:
            raise ValueError("Student range upper bound must be a valid number.") from exc

        if range_value <= 0:
            raise ValueError("Student range upper bound must be greater than zero.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO students (name, grade, range_upper)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    grade = excluded.grade,
                    range_upper = excluded.range_upper,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, float(grade), range_value),
            )
            conn.commit()

        self._sync_students_json()

    def delete_student(self, name: str) -> bool:
        key = name.strip()
        if not key:
            return False

        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM students WHERE name = ?", (key,))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            self._sync_students_json()
        return deleted

    def _sync_students_json(self) -> None:
        records = self.list_students()
        payload = {row["name"]: row["grade"] for row in records}
        self.students_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.students_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_gdp_data(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    country_code AS 'Country Code',
                    country_name AS 'Country Name',
                    year AS Year,
                    gdp AS GDP
                FROM gdp_data
                """,
                conn,
            )

        if not df.empty:
            df["Year"] = pd.to_numeric(df["Year"])
            df["GDP"] = pd.to_numeric(df["GDP"])
        return df