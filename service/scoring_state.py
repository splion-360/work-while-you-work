import json
import sqlite3
import fcntl
from contextlib import closing, contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_inputs (
    application_key TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    source_job_id TEXT,
    job_description TEXT NOT NULL,
    job_description_hash TEXT NOT NULL,
    description_provenance TEXT NOT NULL
        CHECK (description_provenance IN ('original_saved', 'manual_supplied', 'recaptured_current_page')),
    captured_at TEXT NOT NULL,
    extraction_quality TEXT NOT NULL
        CHECK (extraction_quality IN ('complete', 'partial', 'unavailable')),
    resume_version TEXT NOT NULL,
    resume_path TEXT NOT NULL,
    resume_content_hash TEXT NOT NULL,
    role_company_key TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS score_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_key TEXT NOT NULL REFERENCES scoring_inputs(application_key) ON DELETE CASCADE,
    input_fingerprint TEXT NOT NULL UNIQUE,
    scorer_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    deployment_revision TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL CHECK (score >= 0 AND score <= 100),
    band TEXT NOT NULL CHECK (band IN ('Strong', 'Moderate', 'Weak')),
    matched_terms_json TEXT NOT NULL,
    category_breakdown_json TEXT NOT NULL,
    fit_label TEXT,
    fit_probability REAL,
    decision_threshold REAL,
    is_application_time INTEGER NOT NULL DEFAULT 0 CHECK (is_application_time IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scoring_input_snapshots (
    input_fingerprint TEXT PRIMARY KEY,
    application_key TEXT NOT NULL REFERENCES scoring_inputs(application_key) ON DELETE CASCADE,
    canonical_url TEXT NOT NULL,
    source_job_id TEXT,
    job_description TEXT NOT NULL,
    job_description_hash TEXT NOT NULL,
    description_provenance TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    extraction_quality TEXT NOT NULL,
    resume_version TEXT NOT NULL,
    resume_path TEXT NOT NULL,
    resume_content_hash TEXT NOT NULL,
    role_company_key TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS one_application_time_score
ON score_runs(application_key) WHERE is_application_time = 1;

CREATE TABLE IF NOT EXISTS score_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_key TEXT NOT NULL REFERENCES scoring_inputs(application_key) ON DELETE CASCADE,
    idempotency_key TEXT NOT NULL UNIQUE,
    input_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('waiting_application', 'queued', 'running', 'completed', 'failed', 'cancelled')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    next_run_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    lease_owner TEXT,
    lease_until TEXT,
    publishing INTEGER NOT NULL DEFAULT 0 CHECK (publishing IN (0, 1)),
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS runnable_score_jobs
ON score_jobs(status, next_run_at, lease_until);

CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    texts_hash TEXT NOT NULL,
    embeddings_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS score_cache (
    cache_key TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class ScoringStore:
    def __init__(self, path):
        self.path = Path(path)

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def get_cached_score(self, cache_key):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT result_json FROM score_cache WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP",
                (cache_key,),
            ).fetchone()
            return json.loads(row["result_json"]) if row else None

    def save_cached_score(self, cache_key, result, ttl_seconds=300):
        payload = json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM score_cache WHERE expires_at <= CURRENT_TIMESTAMP")
            connection.execute(
                """
                INSERT INTO score_cache (cache_key, result_json, expires_at)
                VALUES (?, ?, datetime('now', ?))
                ON CONFLICT(cache_key) DO UPDATE SET
                    result_json = excluded.result_json,
                    expires_at = excluded.expires_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (cache_key, payload, f"+{int(ttl_seconds)} seconds"),
            )

    @contextmanager
    def publication_lock(self):
        lock_path = self.path.with_suffix(f"{self.path.suffix}.notion.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def initialize(self):
        with closing(self.connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
            job_columns = {row[1] for row in connection.execute("PRAGMA table_info(score_jobs)")}
            if "publishing" not in job_columns:
                connection.execute(
                    "ALTER TABLE score_jobs ADD COLUMN publishing INTEGER NOT NULL DEFAULT 0"
                )
            run_columns = {row[1] for row in connection.execute("PRAGMA table_info(score_runs)")}
            if "deployment_revision" not in run_columns:
                connection.execute(
                    "ALTER TABLE score_runs ADD COLUMN deployment_revision TEXT NOT NULL DEFAULT ''"
                )
            for name, data_type in (
                ("fit_label", "TEXT"),
                ("fit_probability", "REAL"),
                ("decision_threshold", "REAL"),
            ):
                if name not in run_columns:
                    connection.execute(
                        f"ALTER TABLE score_runs ADD COLUMN {name} {data_type}"
                    )
            for table in ("scoring_inputs", "scoring_input_snapshots"):
                columns = {
                    row[1] for row in connection.execute(f"PRAGMA table_info({table})")
                }
                if "role_company_key" not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN role_company_key TEXT NOT NULL DEFAULT ''"
                    )
            connection.execute(
                """
                INSERT OR IGNORE INTO scoring_input_snapshots (
                    input_fingerprint, application_key, canonical_url, source_job_id,
                    job_description, job_description_hash, description_provenance,
                    captured_at, extraction_quality, resume_version, resume_path,
                    resume_content_hash, role_company_key
                )
                SELECT j.input_fingerprint, i.application_key, i.canonical_url, i.source_job_id,
                       i.job_description, i.job_description_hash, i.description_provenance,
                       i.captured_at, i.extraction_quality, i.resume_version, i.resume_path,
                       i.resume_content_hash, i.role_company_key
                FROM score_jobs j JOIN scoring_inputs i USING (application_key)
                """
            )
            connection.execute("PRAGMA user_version = 4")

    def save_scoring_input(self, scoring_input):
        columns = (
            "application_key", "canonical_url", "source_job_id", "job_description",
            "job_description_hash", "description_provenance", "captured_at",
            "extraction_quality", "resume_version", "resume_path", "resume_content_hash",
            "role_company_key",
        )
        values = tuple(
            scoring_input.get(column, "")
            if column == "role_company_key" else scoring_input[column]
            for column in columns
        )
        placeholders = ", ".join("?" for _ in columns)
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                f"INSERT OR IGNORE INTO scoring_inputs ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            return cursor.rowcount == 1

    def get_scoring_input(self, application_key):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM scoring_inputs WHERE application_key = ?",
                (application_key,),
            ).fetchone()
            return dict(row) if row else None

    def delete_application(self, application_key):
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM scoring_inputs WHERE application_key = ?",
                (application_key,),
            )
            return cursor.rowcount == 1

    def rename_application(self, old_key, new_key):
        if old_key == new_key:
            return True
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM scoring_inputs WHERE application_key = ?",
                (old_key,),
            ).fetchone()
            if not row:
                return False
            existing = connection.execute(
                "SELECT 1 FROM scoring_inputs WHERE application_key = ?",
                (new_key,),
            ).fetchone()
            if existing:
                return False
            values = dict(row)
            values["application_key"] = new_key
            columns = tuple(values.keys())
            connection.execute(
                f"""
                INSERT INTO scoring_inputs ({', '.join(columns)})
                VALUES ({', '.join('?' for _ in columns)})
                """,
                tuple(values[column] for column in columns),
            )
            for table in ("score_runs", "scoring_input_snapshots", "score_jobs"):
                connection.execute(
                    f"UPDATE {table} SET application_key = ? WHERE application_key = ?",
                    (new_key, old_key),
                )
            connection.execute(
                "DELETE FROM scoring_inputs WHERE application_key = ?",
                (old_key,),
            )
            return True

    def save_scoring_snapshot(self, input_fingerprint, scoring_input):
        columns = (
            "application_key", "canonical_url", "source_job_id", "job_description",
            "job_description_hash", "description_provenance", "captured_at",
            "extraction_quality", "resume_version", "resume_path", "resume_content_hash",
            "role_company_key",
        )
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                f"""
                INSERT OR IGNORE INTO scoring_input_snapshots
                    (input_fingerprint, {', '.join(columns)})
                VALUES (?, {', '.join('?' for _ in columns)})
                """,
                (
                    input_fingerprint,
                    *(
                        scoring_input.get(column, "")
                        if column == "role_company_key" else scoring_input[column]
                        for column in columns
                    ),
                ),
            )
            return cursor.rowcount == 1

    def get_scoring_snapshot(self, input_fingerprint):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM scoring_input_snapshots WHERE input_fingerprint = ?",
                (input_fingerprint,),
            ).fetchone()
            return dict(row) if row else None

    def list_scoring_snapshots(self):
        with closing(self.connect()) as connection, connection:
            return [
                dict(row) for row in connection.execute(
                    "SELECT * FROM scoring_input_snapshots ORDER BY input_fingerprint"
                )
            ]

    def get_cached_embeddings(self, cache_key):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT embeddings_json FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return json.loads(row["embeddings_json"]) if row else None

    def save_cached_embeddings(self, cache_key, content_hash, model, texts_hash, embeddings):
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (cache_key, content_hash, embedding_model, texts_hash, embeddings_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_key, content_hash, model, texts_hash, json.dumps(embeddings, separators=(",", ":"))),
            )

    def save_score_run(self, result):
        matched_json = json.dumps(result["matched_terms"], ensure_ascii=True, separators=(",", ":"))
        category_json = json.dumps(
            result["category_breakdown"], ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        with closing(self.connect()) as connection, connection:
            application_time = connection.execute(
                "SELECT input_fingerprint FROM score_runs WHERE application_key = ? AND is_application_time = 1",
                (result["application_key"],),
            ).fetchone()
            existing = connection.execute(
                "SELECT id, is_application_time FROM score_runs WHERE input_fingerprint = ?",
                (result["input_fingerprint"],),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE score_runs
                    SET score = ?, band = ?, matched_terms_json = ?, category_breakdown_json = ?,
                        deployment_revision = ?, fit_label = ?, fit_probability = ?,
                        decision_threshold = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        result["score"], result["band"], matched_json, category_json,
                        result.get("deployment_revision", ""), result.get("fit_label"),
                        result.get("fit_probability"), result.get("decision_threshold"),
                        existing["id"],
                    ),
                )
                is_application_time = existing["is_application_time"]
            else:
                is_application_time = 0 if application_time else 1
                connection.execute(
                    """
                    INSERT INTO score_runs (
                        application_key, input_fingerprint, scorer_version, embedding_model,
                        deployment_revision,
                        score, band, matched_terms_json, category_breakdown_json,
                        fit_label, fit_probability, decision_threshold, is_application_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result["application_key"], result["input_fingerprint"], result["scorer_version"],
                        result["embedding_model"], result.get("deployment_revision", ""),
                        result["score"], result["band"], matched_json,
                        category_json, result.get("fit_label"),
                        result.get("fit_probability"), result.get("decision_threshold"),
                        is_application_time,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM score_runs WHERE input_fingerprint = ?",
                (result["input_fingerprint"],),
            ).fetchone()
            return dict(row)

    def list_score_runs(self, application_key):
        with closing(self.connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM score_runs WHERE application_key = ? ORDER BY id",
                (application_key,),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_scoring_inputs(self):
        with closing(self.connect()) as connection, connection:
            return [dict(row) for row in connection.execute("SELECT * FROM scoring_inputs ORDER BY application_key")]

    def list_all_score_runs(self):
        with closing(self.connect()) as connection, connection:
            return [dict(row) for row in connection.execute("SELECT * FROM score_runs ORDER BY id")]

    def get_score_result(self, input_fingerprint):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM score_runs WHERE input_fingerprint = ?",
                (input_fingerprint,),
            ).fetchone()
            if not row:
                return None
            return {
                "application_key": row["application_key"],
                "input_fingerprint": row["input_fingerprint"],
                "scorer_version": row["scorer_version"],
                "embedding_model": row["embedding_model"],
                "deployment_revision": row["deployment_revision"],
                "score": row["score"],
                "band": row["band"],
                "matched_terms": json.loads(row["matched_terms_json"]),
                "category_breakdown": json.loads(row["category_breakdown_json"]),
                "fit_label": row["fit_label"],
                "fit_probability": row["fit_probability"],
                "decision_threshold": row["decision_threshold"],
            }

    def enqueue_score_job(self, application_key, input_fingerprint, status="waiting_application"):
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO score_jobs
                    (application_key, idempotency_key, input_fingerprint, status)
                VALUES (?, ?, ?, ?)
                """,
                (application_key, input_fingerprint, input_fingerprint, status),
            )
            row = connection.execute(
                "SELECT * FROM score_jobs WHERE idempotency_key = ?",
                (input_fingerprint,),
            ).fetchone()
            return dict(row)

    def activate_score_job(self, input_fingerprint):
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE score_jobs
                SET status = 'queued', next_run_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE idempotency_key = ? AND status = 'waiting_application'
                """,
                (input_fingerprint,),
            )
            return cursor.rowcount == 1

    def claim_score_job(self, worker_id, lease_seconds=300):
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM score_jobs
                WHERE (status = 'queued' AND next_run_at <= CURRENT_TIMESTAMP)
                   OR (status = 'running' AND publishing = 0 AND lease_until <= CURRENT_TIMESTAMP)
                ORDER BY next_run_at, id
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            connection.execute(
                """
                UPDATE score_jobs
                SET status = 'running', publishing = 0, lease_owner = ?,
                    lease_until = datetime('now', ?), attempt_count = attempt_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (worker_id, f"+{int(lease_seconds)} seconds", row["id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM score_jobs WHERE id = ?", (row["id"],)
            ).fetchone()
            return dict(claimed)

    def begin_score_publication(self, job_id, worker_id, attempt_count, lease_seconds=300):
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE score_jobs
                SET publishing = 1, lease_until = datetime('now', ?), updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'running' AND publishing = 0
                  AND lease_owner = ? AND attempt_count = ? AND lease_until > CURRENT_TIMESTAMP
                """,
                (f"+{int(lease_seconds)} seconds", job_id, worker_id, attempt_count),
            )
            return cursor.rowcount == 1

    def complete_score_job(self, job_id, worker_id=None, attempt_count=None):
        with closing(self.connect()) as connection, connection:
            ownership = "" if worker_id is None else " AND lease_owner = ? AND attempt_count = ?"
            parameters = (job_id,) if worker_id is None else (job_id, worker_id, attempt_count)
            cursor = connection.execute(
                """
                UPDATE score_jobs
                SET status = 'completed', publishing = 0, lease_owner = NULL, lease_until = NULL,
                    last_error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """ + ownership,
                parameters,
            )
            return cursor.rowcount == 1

    def fail_score_job(
        self, job_id, error, max_attempts=5, retry_delay_seconds=5,
        worker_id=None, attempt_count=None,
    ):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                "SELECT attempt_count, lease_owner FROM score_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                return
            if worker_id is not None and (
                row["lease_owner"] != worker_id or row["attempt_count"] != attempt_count
            ):
                return
            terminal = row["attempt_count"] >= max_attempts
            connection.execute(
                """
                UPDATE score_jobs
                SET status = ?, publishing = 0, next_run_at = datetime('now', ?), lease_owner = NULL,
                    lease_until = NULL, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    "failed" if terminal else "queued",
                    f"+{max(0, int(retry_delay_seconds))} seconds",
                    str(error)[:500],
                    job_id,
                ),
            )

    def list_score_jobs(self):
        with closing(self.connect()) as connection, connection:
            return [dict(row) for row in connection.execute("SELECT * FROM score_jobs ORDER BY id")]

    def list_expired_publications(self):
        with closing(self.connect()) as connection, connection:
            return [
                dict(row) for row in connection.execute(
                    """
                    SELECT * FROM score_jobs
                    WHERE status = 'running' AND publishing = 1
                      AND lease_until <= CURRENT_TIMESTAMP
                    ORDER BY id
                    """
                )
            ]

    def requeue_expired_publication(self, job_id, uncertainty_seconds):
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE score_jobs
                SET status = 'queued', publishing = 0, lease_owner = NULL,
                    lease_until = NULL, next_run_at = CURRENT_TIMESTAMP,
                    last_error = 'Publication outcome was not found in Notion after uncertainty window',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'running' AND publishing = 1
                  AND lease_until <= CURRENT_TIMESTAMP
                  AND updated_at <= datetime('now', ?)
                """,
                (job_id, f"-{int(uncertainty_seconds)} seconds"),
            )
            return cursor.rowcount == 1

    def recover_expired_leases(self):
        with closing(self.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE score_jobs
                SET status = 'queued', publishing = 0, lease_owner = NULL, lease_until = NULL,
                    next_run_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE status = 'running' AND publishing = 0
                  AND lease_until <= CURRENT_TIMESTAMP
                """
            )
            return cursor.rowcount

    def record_worker_heartbeat(self, worker_id):
        with closing(self.connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO worker_heartbeats (worker_id, updated_at)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(worker_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (worker_id,),
            )

    def worker_is_healthy(self, max_age_seconds):
        with closing(self.connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT MIN((julianday('now') - julianday(updated_at)) * 86400) AS age_seconds
                FROM worker_heartbeats
                """
            ).fetchone()
            return row["age_seconds"] is not None and row["age_seconds"] <= max_age_seconds

    def is_healthy(self):
        with closing(self.connect()) as connection, connection:
            return connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
