import os
import json
import psycopg
from psycopg.rows import dict_row

def _get_db_conn():
    db_host = os.getenv("DB_HOST")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASS")
    
    if not db_host:
        raise Exception("Database configuration missing (DB_HOST not set)")
        
    return psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass,
        row_factory=dict_row
    )

def _execute_query(query: str, params: tuple = ()):
    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if query.strip().upper().startswith("SELECT") or "RETURNING" in query.upper():
                return cur.fetchall()
            conn.commit()
            return []

def get_tickets_by_status(status: str) -> dict:
    try:
        tickets = _execute_query("SELECT * FROM tickets WHERE status ILIKE %s ORDER BY creation_time DESC", (status,))
        return {"success": True, "tickets": tickets, "count": len(tickets)}
    except Exception as e:
        return {"error": str(e), "tickets": []}

def get_tickets_by_priority(priority: str) -> dict:
    try:
        tickets = _execute_query("SELECT * FROM tickets WHERE priority ILIKE %s ORDER BY creation_time DESC", (priority,))
        return {"success": True, "tickets": tickets, "count": len(tickets)}
    except Exception as e:
        return {"error": str(e), "tickets": []}

def get_ticket_by_id(ticket_id: str) -> dict:
    try:
        tickets = _execute_query("SELECT * FROM tickets WHERE ticket_id = %s", (int(ticket_id),))
        ticket = tickets[0] if tickets else None
        return {"success": True, "ticket": ticket}
    except Exception as e:
        return {"error": str(e), "ticket": None}

def get_tickets_by_assignee(assignee: str) -> dict:
    try:
        tickets = _execute_query("SELECT * FROM tickets WHERE assignee ILIKE %s ORDER BY creation_time DESC", (f"%{assignee}%",))
        return {"success": True, "tickets": tickets, "count": len(tickets)}
    except Exception as e:
        return {"error": str(e), "tickets": []}

def db_create_ticket(title: str, description: str, assignee: str = "", priority: str = "P3 - Low", status: str = "Open") -> dict:
    try:
        tickets = _execute_query(
            "INSERT INTO tickets (title, description, assignee, priority, status) VALUES (%s, %s, %s, %s, %s) RETURNING ticket_id",
            (title, description, assignee, priority, status)
        )
        ticket_id = tickets[0]['ticket_id'] if tickets else None
        return {"success": True, "ticket_id": str(ticket_id), "message": "Ticket created successfully"}
    except Exception as e:
        return {"error": str(e), "ticket_id": None}

def update_ticket_status(ticket_id: str, status: str) -> dict:
    try:
        _execute_query("UPDATE tickets SET status = %s, updated_time = CURRENT_TIMESTAMP WHERE ticket_id = %s", (status, int(ticket_id)))
        return {"success": True, "message": f"Ticket {ticket_id} status updated to {status}"}
    except Exception as e:
        return {"error": str(e), "success": False}

def update_ticket_priority(ticket_id: str, priority: str) -> dict:
    try:
        _execute_query("UPDATE tickets SET priority = %s, updated_time = CURRENT_TIMESTAMP WHERE ticket_id = %s", (priority, int(ticket_id)))
        return {"success": True, "message": f"Ticket {ticket_id} priority updated to {priority}"}
    except Exception as e:
        return {"error": str(e), "success": False}

def get_tickets_by_date_range(start_date: str, end_date: str, date_field: str = "creation_time") -> dict:
    try:
        query = f"SELECT * FROM tickets WHERE {date_field} >= %s AND {date_field} <= %s ORDER BY {date_field} DESC"
        tickets = _execute_query(query, (start_date, end_date))
        return {"success": True, "tickets": tickets, "count": len(tickets)}
    except Exception as e:
        return {"error": str(e), "tickets": []}

def update_ticket_embedding(ticket_id: str, embedding: str):
    try:
        _execute_query("UPDATE tickets SET embedding = %s::vector WHERE ticket_id = %s", (embedding, int(ticket_id)))
    except Exception as e:
        print(f"Failed to update embedding: {e}")

def search_tickets_vector(embedding_vector: str, limit: int = 5) -> dict:
    try:
        tickets = _execute_query("SELECT ticket_id, title, description, status, priority, assignee, 1 - (embedding <=> %s::vector) as similarity FROM tickets WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s", (embedding_vector, embedding_vector, limit))
        return {"success": True, "tickets": tickets, "count": len(tickets)}
    except Exception as e:
        return {"error": str(e), "tickets": []}
