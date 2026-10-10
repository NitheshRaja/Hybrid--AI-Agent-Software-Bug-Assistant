import os
import json
import sqlite3
from typing import Tuple, List, Dict, Any, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets_fallback.db")

SAMPLE_TICKETS = [
    ("Login Page Freezes After Multiple Failed Attempts", "Users are reporting that after 3 failed login attempts, the login page becomes unresponsive and requires a refresh. No specific error message is displayed.", "samuel.green@example.com", "P0 - Critical", "Open"),
    ("Dashboard Sales Widget Intermittent Data Loading Failure", "The \"Sales Overview\" widget on the main dashboard intermittently shows a loading spinner but no data. Primarily affects Chrome browser users.", "maria.rodriguez@example.com", "P1 - High", "In Progress"),
    ("Broken Link in Footer - Privacy Policy", "The \"Privacy Policy\" hyperlink located in the website footer leads to a 404 \"Page Not Found\" error.", "maria.rodriguez@example.com", "P3 - Low", "Resolved"),
    ("UI Misalignment on Mobile Landscape View (iOS)", "On specific iOS devices (e.g., iPhone 14 models), the top navigation bar shifts downwards when the device is viewed in landscape orientation, obscuring content.", "maria.rodriguez@example.com", "P2 - Medium", "In Progress"),
    ("Critical XZ Utils Backdoor Detected in Core Dependency (CVE-2024-3094)", "Urgent: A sophisticated supply chain compromise (CVE-2024-3094) has been identified in XZ Utils versions 5.6.0 and 5.6.1. This malicious code potentially allows unauthorized remote SSH access by modifying liblzma. Immediate investigation and action required for affected Linux/Unix systems and services relying on XZ Utils.", "frank.white@example.com", "P0 - Critical", "Open"),
    ("Database Connection Timeouts During Peak Usage", "The application is experiencing frequent database connection timeouts, particularly during peak hours (10 AM - 12 PM EDT), affecting all users and causing service interruptions.", "frank.white@example.com", "P1 - High", "Open"),
    ("Export to PDF Truncates Long Text Fields in Reports", "When generating PDF exports of reports containing extensive text fields, the text is abruptly cut off at the end of the page instead of wrapping or continuing to the next page.", "samuel.green@example.com", "P1 - High", "Open"),
    ("Search Filter \"Date Range\" Not Applying Correctly", "The \"Date Range\" filter on the search results page does not filter records accurately; results outside the specified date range are still displayed.", "samuel.green@example.com", "P2 - Medium", "Resolved"),
    ("Typo in Error Message: \"Unathorized Access\"", "The error message displayed when a user attempts an unauthorized action reads \"Unathorized Access\" instead of \"Unauthorized Access.\"", "maria.rodriguez@example.com", "P3 - Low", "Resolved"),
    ("Intermittent File Upload Failures for Large Files", "Users are intermittently reporting that file uploads fail without a clear error message or explanation, especially for files exceeding 10MB in size.", "frank.white@example.com", "P1 - High", "Open")
]

def _init_sqlite_db():
    """Ensure the local fallback SQLite database has the tickets table and initial sample tickets."""
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                assignee TEXT,
                priority TEXT,
                status TEXT DEFAULT 'Open',
                creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("SELECT COUNT(*) FROM tickets")
        count = cur.fetchone()[0]
        if count == 0:
            cur.executemany(
                "INSERT INTO tickets (title, description, assignee, priority, status) VALUES (?, ?, ?, ?, ?)",
                SAMPLE_TICKETS
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Fallback SQLite init error: {e}")

# Initialize local fallback database
_init_sqlite_db()


def _get_db_conn():
    if not psycopg:
        raise Exception("psycopg library not available")

    # Direct connection string (common in Supabase / Neon / Heroku / HF)
    db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or os.getenv("SUPABASE_DB_URL")
    if db_url:
        return psycopg.connect(db_url, row_factory=dict_row, connect_timeout=4)

    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD")
    
    if not db_host:
        raise Exception("Database configuration missing: DB_HOST (or DATABASE_URL) environment variable is not set.")
    
    # Supabase / cloud DBs need SSL, local dev usually doesn't
    sslmode = os.getenv("DB_SSLMODE")
    if not sslmode:
        sslmode = "disable" if db_host in ("127.0.0.1", "localhost") else "require"
    
    conninfo = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_pass} sslmode={sslmode} connect_timeout=4"
    
    return psycopg.connect(
        conninfo,
        row_factory=dict_row
    )


def _execute_sqlite_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    _init_sqlite_db()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sqlite_query = query.replace("%s", "?").replace("ILIKE", "LIKE")
    
    cur.execute(sqlite_query, params)

    if sqlite_query.strip().upper().startswith("SELECT") or "RETURNING" in sqlite_query.upper():
        rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        conn.close()
        return rows

    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    if "INSERT" in sqlite_query.upper():
        return [{"ticket_id": last_id}]
    return []


def _execute_query(query: str, params: tuple = ()) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Execute a query against PostgreSQL first, falling back to local SQLite if unavailable."""
    # Try PostgreSQL first
    try:
        with _get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if query.strip().upper().startswith("SELECT") or "RETURNING" in query.upper():
                    return cur.fetchall(), False, None
                conn.commit()
                return [], False, None
    except Exception as pg_err:
        print(f"[Database] PostgreSQL unavailable ({pg_err}). Falling back to local SQLite database.")
        try:
            sqlite_rows = _execute_sqlite_query(query, params)
            return sqlite_rows, True, str(pg_err)
        except Exception as sqlite_err:
            raise Exception(f"PostgreSQL error: {pg_err} | SQLite error: {sqlite_err}")


def get_tickets_by_status(status: str) -> dict:
    try:
        tickets, is_fallback, pg_err = _execute_query(
            "SELECT * FROM tickets WHERE status ILIKE %s ORDER BY creation_time DESC",
            (status,)
        )
        res = {"success": True, "tickets": tickets, "count": len(tickets)}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "tickets": []}


def get_tickets_by_priority(priority: str) -> dict:
    try:
        tickets, is_fallback, pg_err = _execute_query(
            "SELECT * FROM tickets WHERE priority ILIKE %s ORDER BY creation_time DESC",
            (priority,)
        )
        res = {"success": True, "tickets": tickets, "count": len(tickets)}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "tickets": []}


def get_ticket_by_id(ticket_id: str) -> dict:
    try:
        tickets, is_fallback, pg_err = _execute_query(
            "SELECT * FROM tickets WHERE ticket_id = %s",
            (int(ticket_id),)
        )
        ticket = tickets[0] if tickets else None
        res = {"success": True, "ticket": ticket}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "ticket": None}


def get_tickets_by_assignee(assignee: str) -> dict:
    try:
        tickets, is_fallback, pg_err = _execute_query(
            "SELECT * FROM tickets WHERE assignee ILIKE %s ORDER BY creation_time DESC",
            (f"%{assignee}%",)
        )
        res = {"success": True, "tickets": tickets, "count": len(tickets)}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "tickets": []}


def db_create_ticket(title: str, description: str, assignee: str = "", priority: str = "P3 - Low", status: str = "Open") -> dict:
    try:
        tickets, is_fallback, pg_err = _execute_query(
            "INSERT INTO tickets (title, description, assignee, priority, status) VALUES (%s, %s, %s, %s, %s) RETURNING ticket_id",
            (title, description, assignee, priority, status)
        )
        ticket_id = tickets[0]['ticket_id'] if tickets else None
        res = {"success": True, "ticket_id": str(ticket_id), "message": "Ticket created successfully"}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "ticket_id": None}


def update_ticket_status(ticket_id: str, status: str) -> dict:
    try:
        _, is_fallback, pg_err = _execute_query(
            "UPDATE tickets SET status = %s, updated_time = CURRENT_TIMESTAMP WHERE ticket_id = %s",
            (status, int(ticket_id))
        )
        res = {"success": True, "message": f"Ticket {ticket_id} status updated to {status}"}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "success": False}


def update_ticket_priority(ticket_id: str, priority: str) -> dict:
    try:
        _, is_fallback, pg_err = _execute_query(
            "UPDATE tickets SET priority = %s, updated_time = CURRENT_TIMESTAMP WHERE ticket_id = %s",
            (priority, int(ticket_id))
        )
        res = {"success": True, "message": f"Ticket {ticket_id} priority updated to {priority}"}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "success": False}


def get_tickets_by_date_range(start_date: str, end_date: str, date_field: str = "creation_time") -> dict:
    try:
        query = f"SELECT * FROM tickets WHERE {date_field} >= %s AND {date_field} <= %s ORDER BY {date_field} DESC"
        tickets, is_fallback, pg_err = _execute_query(query, (start_date, end_date))
        res = {"success": True, "tickets": tickets, "count": len(tickets)}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        return {"error": str(e), "tickets": []}


def update_ticket_embedding(ticket_id: str, embedding: str):
    try:
        _execute_query("UPDATE tickets SET embedding = %s::vector WHERE ticket_id = %s", (embedding, int(ticket_id)))
    except Exception as e:
        print(f"Failed to update embedding: {e}")


def search_tickets_vector(embedding_vector: str, limit: int = 5) -> dict:
    try:
        tickets, is_fallback, pg_err = _execute_query(
            "SELECT ticket_id, title, description, status, priority, assignee, 1 - (embedding <=> %s::vector) as similarity FROM tickets WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s",
            (embedding_vector, embedding_vector, limit)
        )
        res = {"success": True, "tickets": tickets, "count": len(tickets)}
        if is_fallback:
            res["is_fallback"] = True
            res["fallback_reason"] = pg_err
        return res
    except Exception as e:
        try:
            sqlite_tickets = _execute_sqlite_query("SELECT ticket_id, title, description, status, priority, assignee, 0.9 as similarity FROM tickets LIMIT ?", (limit,))
            return {"success": True, "tickets": sqlite_tickets, "count": len(sqlite_tickets), "is_fallback": True}
        except Exception:
            return {"error": str(e), "tickets": []}

