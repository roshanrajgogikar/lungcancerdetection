"""Account storage for the Streamlit app.

SQLite is convenient for local development. Configure DATABASE_URL to use a
persistent PostgreSQL database when deploying the app publicly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import bcrypt
from sqlalchemy import Column, DateTime, Integer, LargeBinary, MetaData, String, Table, create_engine, func, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
metadata = MetaData()
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("email", String(254), nullable=False, unique=True, index=True),
    Column("password_hash", LargeBinary(60), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


def build_engine(database_url: str | None = None) -> Engine:
    if database_url:
        url = database_url.strip()
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        engine = create_engine(url, pool_pre_ping=True)
    else:
        INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
        sqlite_path = (INSTANCE_DIR / "accounts.db").as_posix()
        engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
    metadata.create_all(engine)
    return engine


def validate_registration(email: str, password: str, confirm_password: str) -> tuple[str | None, str | None]:
    clean_email = email.strip().lower()
    if len(clean_email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", clean_email):
        return None, "Enter a valid email address."
    if len(password) < 8:
        return None, "Use a password with at least 8 characters."
    if len(password.encode("utf-8")) > 72:
        return None, "Use a password with no more than 72 UTF-8 bytes."
    if password != confirm_password:
        return None, "The passwords do not match."
    return clean_email, None


def register_user(engine: Engine, email: str, password: str) -> tuple[bool, str]:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    try:
        with engine.begin() as connection:
            connection.execute(insert(users).values(email=email, password_hash=password_hash))
        return True, "Account created. Sign in with your new credentials."
    except IntegrityError:
        return False, "An account with this email already exists."


def authenticate_user(engine: Engine, email: str, password: str) -> bool:
    with engine.connect() as connection:
        row: Any = connection.execute(select(users.c.password_hash).where(users.c.email == email)).first()
    if row is None:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), bytes(row.password_hash))
    except (ValueError, TypeError):
        return False
