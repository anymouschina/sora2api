"""SQLite database layer for Gemini/Flow token pool.

Uses the same sqlite file as Sora by default, but stores data in dedicated
tables prefixed with `gemini_` to keep token pools separated.
"""

from __future__ import annotations

import aiosqlite
import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    GeminiToken,
    GeminiTokenStats,
    GeminiTask,
    GeminiRequestLog,
    GeminiProject,
    GeminiCaptchaConfig,
)


class GeminiDatabase:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "hancat.db")
        self.db_path = db_path

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    st TEXT UNIQUE NOT NULL,
                    at TEXT,
                    at_expires TIMESTAMP,
                    email TEXT NOT NULL,
                    name TEXT,
                    remark TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    credits INTEGER DEFAULT 0,
                    user_paygate_tier TEXT,
                    current_project_id TEXT,
                    current_project_name TEXT,
                    image_enabled BOOLEAN DEFAULT 1,
                    video_enabled BOOLEAN DEFAULT 1,
                    image_concurrency INTEGER DEFAULT -1,
                    video_concurrency INTEGER DEFAULT -1,
                    ban_reason TEXT,
                    banned_at TIMESTAMP
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE NOT NULL,
                    token_id INTEGER NOT NULL,
                    project_name TEXT NOT NULL,
                    tool_name TEXT DEFAULT 'PINHOLE',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES gemini_tokens(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_token_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id INTEGER NOT NULL,
                    image_count INTEGER DEFAULT 0,
                    video_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    last_success_at TIMESTAMP,
                    last_error_at TIMESTAMP,
                    today_image_count INTEGER DEFAULT 0,
                    today_video_count INTEGER DEFAULT 0,
                    today_error_count INTEGER DEFAULT 0,
                    today_date DATE,
                    consecutive_error_count INTEGER DEFAULT 0,
                    FOREIGN KEY (token_id) REFERENCES gemini_tokens(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    token_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing',
                    progress INTEGER DEFAULT 0,
                    result_urls TEXT,
                    error_message TEXT,
                    scene_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES gemini_tokens(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id INTEGER,
                    operation TEXT NOT NULL,
                    request_body TEXT,
                    response_body TEXT,
                    status_code INTEGER NOT NULL,
                    duration FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES gemini_tokens(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS gemini_captcha_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    captcha_method TEXT DEFAULT 'browser',
                    yescaptcha_api_key TEXT DEFAULT '',
                    yescaptcha_base_url TEXT DEFAULT 'https://api.yescaptcha.com',
                    browser_proxy_enabled BOOLEAN DEFAULT 0,
                    browser_proxy_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ensure default gemini_captcha_config row exists
            cursor = await db.execute("SELECT COUNT(*) FROM gemini_captcha_config WHERE id = 1")
            count = await cursor.fetchone()
            if count and count[0] == 0:
                await db.execute("INSERT INTO gemini_captcha_config (id) VALUES (1)")

            # Best-effort migration from legacy `captcha_config` table (older merged builds)
            try:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='captcha_config'"
                )
                legacy = await cursor.fetchone()
                if legacy:
                    db.row_factory = aiosqlite.Row
                    legacy_row = await (await db.execute("SELECT * FROM captcha_config WHERE id = 1")).fetchone()
                    if legacy_row:
                        await db.execute(
                            """
                            UPDATE gemini_captcha_config
                            SET captcha_method = ?,
                                yescaptcha_api_key = ?,
                                yescaptcha_base_url = ?,
                                browser_proxy_enabled = ?,
                                browser_proxy_url = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = 1
                            """,
                            (
                                legacy_row.get("captcha_method"),
                                legacy_row.get("yescaptcha_api_key") or "",
                                legacy_row.get("yescaptcha_base_url") or "https://api.yescaptcha.com",
                                legacy_row.get("browser_proxy_enabled") or 0,
                                legacy_row.get("browser_proxy_url"),
                            ),
                        )
            except Exception:
                pass

            await db.commit()

    # ---------- Captcha config ----------
    async def get_captcha_config(self) -> GeminiCaptchaConfig:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Prefer namespaced table
            try:
                cursor = await db.execute("SELECT * FROM gemini_captcha_config WHERE id = 1")
                row = await cursor.fetchone()
                if row:
                    return GeminiCaptchaConfig(**dict(row))
            except Exception:
                row = None

            # Fallback legacy table
            try:
                cursor = await db.execute("SELECT * FROM captcha_config WHERE id = 1")
                row = await cursor.fetchone()
                if row:
                    return GeminiCaptchaConfig(**dict(row))
            except Exception:
                pass

            return GeminiCaptchaConfig()

    async def update_captcha_config(
        self,
        captcha_method: str,
        yescaptcha_api_key: str,
        yescaptcha_base_url: str,
        browser_proxy_enabled: bool,
        browser_proxy_url: Optional[str],
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE gemini_captcha_config
                SET captcha_method = ?,
                    yescaptcha_api_key = ?,
                    yescaptcha_base_url = ?,
                    browser_proxy_enabled = ?,
                    browser_proxy_url = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
                """,
                (
                    captcha_method,
                    yescaptcha_api_key or "",
                    yescaptcha_base_url or "https://api.yescaptcha.com",
                    1 if browser_proxy_enabled else 0,
                    browser_proxy_url if browser_proxy_url else None,
                ),
            )

            # Best-effort keep legacy table in sync if present
            try:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='captcha_config'"
                )
                legacy = await cursor.fetchone()
                if legacy:
                    await db.execute(
                        """
                        UPDATE captcha_config
                        SET captcha_method = ?,
                            yescaptcha_api_key = ?,
                            yescaptcha_base_url = ?,
                            browser_proxy_enabled = ?,
                            browser_proxy_url = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = 1
                        """,
                        (
                            captcha_method,
                            yescaptcha_api_key or "",
                            yescaptcha_base_url or "https://api.yescaptcha.com",
                            1 if browser_proxy_enabled else 0,
                            browser_proxy_url if browser_proxy_url else None,
                        ),
                    )
            except Exception:
                pass

            await db.commit()

    # ---------- Token CRUD ----------
    async def get_token(self, token_id: int) -> Optional[GeminiToken]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM gemini_tokens WHERE id = ?", (token_id,))
            row = await cursor.fetchone()
            return GeminiToken(**dict(row)) if row else None

    async def get_token_by_st(self, st: str) -> Optional[GeminiToken]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM gemini_tokens WHERE st = ?", (st,))
            row = await cursor.fetchone()
            return GeminiToken(**dict(row)) if row else None

    async def add_token(self, token: GeminiToken) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO gemini_tokens (
                    st, at, at_expires, email, name, remark, is_active,
                    credits, user_paygate_tier, current_project_id, current_project_name,
                    image_enabled, video_enabled, image_concurrency, video_concurrency,
                    ban_reason, banned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token.st,
                    token.at,
                    token.at_expires,
                    token.email or "",
                    token.name or "",
                    token.remark,
                    1 if token.is_active else 0,
                    token.credits or 0,
                    token.user_paygate_tier,
                    token.current_project_id,
                    token.current_project_name,
                    1 if token.image_enabled else 0,
                    1 if token.video_enabled else 0,
                    token.image_concurrency if token.image_concurrency is not None else -1,
                    token.video_concurrency if token.video_concurrency is not None else -1,
                    token.ban_reason,
                    token.banned_at,
                ),
            )
            await db.commit()
            token_id = cursor.lastrowid
            await db.execute("INSERT INTO gemini_token_stats (token_id) VALUES (?)", (token_id,))
            await db.commit()
            return token_id

    async def update_token(self, token_id: int, **fields):
        if not fields:
            return
        allowed = {
            "st",
            "at",
            "at_expires",
            "email",
            "name",
            "remark",
            "is_active",
            "credits",
            "user_paygate_tier",
            "current_project_id",
            "current_project_name",
            "image_enabled",
            "video_enabled",
            "image_concurrency",
            "video_concurrency",
            "ban_reason",
            "banned_at",
        }
        update_fields = {k: v for k, v in fields.items() if k in allowed}
        if not update_fields:
            return
        sets = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [token_id]
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"UPDATE gemini_tokens SET {sets} WHERE id = ?", values)
            await db.commit()

    async def delete_token(self, token_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM gemini_token_stats WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM gemini_projects WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM gemini_tokens WHERE id = ?", (token_id,))
            await db.commit()

    async def get_all_tokens(self) -> List[GeminiToken]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM gemini_tokens ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [GeminiToken(**dict(r)) for r in rows]

    async def get_active_tokens(self) -> List[GeminiToken]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM gemini_tokens
                WHERE is_active = 1
                AND ban_reason IS NULL
                ORDER BY last_used_at ASC NULLS FIRST
                """
            )
            rows = await cursor.fetchall()
            return [GeminiToken(**dict(r)) for r in rows]

    async def update_token_usage(self, token_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE gemini_tokens
                SET last_used_at = CURRENT_TIMESTAMP, use_count = use_count + 1
                WHERE id = ?
                """,
                (token_id,),
            )
            await db.commit()

    # ---------- Projects ----------
    async def add_project(self, project: GeminiProject) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT OR IGNORE INTO gemini_projects (project_id, token_id, project_name, tool_name, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.token_id,
                    project.project_name,
                    project.tool_name or "PINHOLE",
                    1 if project.is_active else 0,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    # ---------- Stats ----------
    async def get_token_stats(self, token_id: int) -> Optional[GeminiTokenStats]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM gemini_token_stats WHERE token_id = ?", (token_id,))
            row = await cursor.fetchone()
            return GeminiTokenStats(**dict(row)) if row else None

    async def increment_token_stats(self, token_id: int, *, image: bool = False, video: bool = False, success: bool = False, error: bool = False):
        today = date.today().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # Ensure today_date set and reset daily counters when date changes
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT today_date FROM gemini_token_stats WHERE token_id = ?", (token_id,))
            row = await cursor.fetchone()
            if row and row["today_date"] != today:
                await db.execute(
                    """
                    UPDATE gemini_token_stats
                    SET today_date = ?, today_image_count = 0, today_video_count = 0, today_error_count = 0
                    WHERE token_id = ?
                    """,
                    (today, token_id),
                )

            updates: List[str] = []
            if image:
                updates += ["image_count = image_count + 1", "today_image_count = today_image_count + 1"]
            if video:
                updates += ["video_count = video_count + 1", "today_video_count = today_video_count + 1"]
            if success:
                updates += ["success_count = success_count + 1", "last_success_at = CURRENT_TIMESTAMP", "consecutive_error_count = 0"]
            if error:
                updates += ["error_count = error_count + 1", "today_error_count = today_error_count + 1", "last_error_at = CURRENT_TIMESTAMP", "consecutive_error_count = consecutive_error_count + 1"]
            if not updates:
                return

            await db.execute(
                f"UPDATE gemini_token_stats SET {', '.join(updates)} WHERE token_id = ?",
                (token_id,),
            )
            await db.commit()

    async def get_consecutive_error_count(self, token_id: int) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT consecutive_error_count FROM gemini_token_stats WHERE token_id = ?",
                (token_id,),
            )
            row = await cursor.fetchone()
            try:
                return int(row[0]) if row and row[0] is not None else 0
            except Exception:
                return 0

    async def get_error_ban_threshold(self) -> int:
        """Reuse Sora admin_config threshold if present."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT error_ban_threshold FROM admin_config WHERE id = 1"
                )
                row = await cursor.fetchone()
                if row and row[0] is not None:
                    return int(row[0])
        except Exception:
            pass
        return 3

    async def reset_consecutive_errors(self, token_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE gemini_token_stats SET consecutive_error_count = 0 WHERE token_id = ?",
                (token_id,),
            )
            await db.commit()

    # ---------- Tasks ----------
    async def create_task(self, task: GeminiTask):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO gemini_tasks (task_id, token_id, model, prompt, status, progress, result_urls, error_message, scene_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.token_id,
                    task.model,
                    task.prompt,
                    task.status,
                    task.progress,
                    json.dumps(task.result_urls) if task.result_urls is not None else None,
                    task.error_message,
                    task.scene_id,
                ),
            )
            await db.commit()

    async def update_task(self, task_id: str, **fields):
        allowed = {"status", "progress", "result_urls", "error_message", "completed_at"}
        update_fields = {k: v for k, v in fields.items() if k in allowed}
        if "result_urls" in update_fields and update_fields["result_urls"] is not None:
            update_fields["result_urls"] = json.dumps(update_fields["result_urls"])
        if not update_fields:
            return
        sets = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [task_id]
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"UPDATE gemini_tasks SET {sets} WHERE task_id = ?", values)
            await db.commit()

    # ---------- Logs ----------
    async def add_request_log(self, log: GeminiRequestLog):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO gemini_request_logs (token_id, operation, request_body, response_body, status_code, duration)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    log.token_id,
                    log.operation,
                    log.request_body,
                    log.response_body,
                    log.status_code,
                    log.duration,
                ),
            )
            await db.commit()

    async def get_request_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT l.*, t.email as token_email
                FROM gemini_request_logs l
                LEFT JOIN gemini_tokens t ON l.token_id = t.id
                ORDER BY l.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
