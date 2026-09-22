"""
setup_db.py — One-time database setup script for Agent OTG
===========================================================

Run this ONCE before starting the application:
    py setup_db.py

What it does:
  1. Reads credentials from db_config.json
  2. Connects to PostgreSQL as the configured user
  3. Creates the 'agent_otg' database if it doesn't exist
  4. Creates the required tables (agent_sessions, agent_messages)
  5. Verifies the setup with a test insert + rollback

Requirements:
  - PostgreSQL must be running on localhost
  - The user in db_config.json must have CREATE DATABASE privilege
    (or the database must already exist)
  - Edit db_config.json with your actual password before running this script
"""

import json
import os
import sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "db_config.json")

# ── ANSI colours ──────────────────────────────────────────────────────────────
def _green(s):  return f"\033[92m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _cyan(s):   return f"\033[96m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"


def load_config():
    if not os.path.isfile(_CONFIG_PATH):
        return None
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("password") == "your_password_here":
        print(_yellow("[WARNING] db_config.json still has the placeholder password."))
        print(_yellow("          Please edit db_config.json and set your real PostgreSQL password."))
        return None
    return cfg


def create_database(cfg: dict):
    """Connect to the 'postgres' maintenance DB and create agent_otg if needed."""
    import psycopg2
    target_db = cfg.get("dbname", "agent_otg")
    print(f"  Checking if database {_cyan(repr(target_db))} exists...")

    try:
        # Connect to the maintenance 'postgres' database
        conn = psycopg2.connect(
            host     = cfg.get("host", "localhost"),
            port     = int(cfg.get("port", 5432)),
            dbname   = "postgres",              # always exists
            user     = cfg.get("user", "postgres"),
            password = cfg.get("password", ""),
            connect_timeout = 5,
        )
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        exists = cur.fetchone()

        if exists:
            print(f"  {_green('OK')} Database {repr(target_db)} already exists.")
        else:
            cur.execute(f'CREATE DATABASE "{target_db}"')
            print(f"  {_green('CREATED')} Database {repr(target_db)} created.")

        cur.close()
        conn.close()
    except psycopg2.OperationalError as exc:
        print(_red(f"\n[ERROR] Could not connect to PostgreSQL: {exc}"))
        print(_yellow("  Ensure PostgreSQL is running and your credentials in db_config.json are correct."))
        sys.exit(1)


def create_tables(cfg: dict):
    """Connect to agent_otg and create the required tables."""
    import psycopg2
    print(f"  Creating tables in {_cyan(repr(cfg.get('dbname', 'agent_otg')))}...")

    conn = psycopg2.connect(
        host     = cfg.get("host", "localhost"),
        port     = int(cfg.get("port", 5432)),
        dbname   = cfg.get("dbname", "agent_otg"),
        user     = cfg.get("user", "postgres"),
        password = cfg.get("password", ""),
        connect_timeout = 5,
    )
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_sessions (
            id           SERIAL PRIMARY KEY,
            session_name VARCHAR(255) UNIQUE NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    print(f"    {_green('OK')} agent_sessions")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_messages (
            id           SERIAL PRIMARY KEY,
            session_name VARCHAR(255) NOT NULL
                            REFERENCES agent_sessions(session_name)
                            ON DELETE CASCADE,
            role         VARCHAR(50)  NOT NULL,
            content      TEXT         NOT NULL,
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        );
    """)
    print(f"    {_green('OK')} agent_messages")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON agent_messages(session_name, created_at);
    """)
    print(f"    {_green('OK')} idx_messages_session (index)")

    conn.commit()
    cur.close()
    conn.close()


def verify_setup(cfg: dict):
    """Quick end-to-end sanity check with a rollback."""
    import psycopg2
    print("  Verifying with a test transaction (rolled back)...")
    conn = psycopg2.connect(
        host     = cfg.get("host", "localhost"),
        port     = int(cfg.get("port", 5432)),
        dbname   = cfg.get("dbname", "agent_otg"),
        user     = cfg.get("user", "postgres"),
        password = cfg.get("password", ""),
        connect_timeout = 5,
    )
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO agent_sessions (session_name) VALUES (%s)",
            ("_setup_test_",)
        )
        cur.execute(
            "INSERT INTO agent_messages (session_name, role, content) VALUES (%s, %s, %s)",
            ("_setup_test_", "user", "setup verification message")
        )
        conn.rollback()   # don't keep the test data
        print(f"  {_green('OK')} Test insert + rollback succeeded.")
    except Exception as exc:
        conn.rollback()
        print(_red(f"  [ERROR] Verification failed: {exc}"))
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


def main():
    print(_bold("\n===  Agent OTG — Database Setup  ===\n"))

    cfg = load_config()
    if cfg is None:
        print(_cyan("No PostgreSQL configuration (db_config.json) found or configured."))
        print("Agent OTG uses built-in SQLite (agent_otg.db) as the primary self-contained database.")
        print("Initializing SQLite database...")
        try:
            import db
            if db.init_db():
                print(f"  {_green('OK')} SQLite database initialized at {_cyan(str(db._DB_PATH))}")
            else:
                print(_red(f"  [ERROR] SQLite initialization failed: {db.get_error()}"))
        except Exception as e:
            print(_red(f"  [ERROR] Failed to initialize SQLite: {e}"))
            sys.exit(1)

        print()
        print(_green(_bold("Setup complete! You can now run:")))
        print(f"  {_cyan('python ask.py')}                               (terminal client)")
        print(f"  {_cyan('python -m uvicorn main:app --reload --port 8000')} (API server)\n")
        return

    print(f"  PostgreSQL config loaded from: {_cyan(_CONFIG_PATH)}")
    print(f"  Host:   {cfg.get('host')}:{cfg.get('port')}")
    print(f"  User:   {cfg.get('user')}")
    print(f"  DB:     {cfg.get('dbname')}\n")

    print(_bold("Step 1: Create database"))
    create_database(cfg)

    print(_bold("\nStep 2: Create tables"))
    create_tables(cfg)

    print(_bold("\nStep 3: Verify"))
    verify_setup(cfg)

    print()
    print(_green(_bold("Setup complete! You can now run:")))
    print(f"  {_cyan('python ask.py')}                               (terminal client)")
    print(f"  {_cyan('python -m uvicorn main:app --reload --port 8000')} (API server)\n")


if __name__ == "__main__":
    main()
