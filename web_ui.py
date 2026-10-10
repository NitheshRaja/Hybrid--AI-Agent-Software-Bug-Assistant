#!/usr/bin/env python
"""
Unified Web UI for Hybrid Software Bug Assistant
- Simple queries → Local Gemma-2B (fast, free)
- Complex queries → Cloud Gemini + Tools (powerful)
- Single interface for both models
"""

import os
import re
import sys
import json
import queue
import base64
from datetime import datetime
from typing import Optional

# Load .env FIRST before anything else
from dotenv import load_dotenv
load_dotenv(override=True)

# Set defaults if not in .env
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
os.environ.setdefault("SKIP_AGENT_IMPORT", "TRUE")

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string, request, jsonify, stream_with_context, Response, make_response

from software_bug_assistant.tools.local_tools import (
    backfill_missing_embeddings,
    get_ticket_creator,
    query_stackexchange,
)
from software_bug_assistant.hybrid_agent import HybridAgent, QueryClassifier, classify_query
import threading
from pathlib import Path

# Application and Agent globals
app = Flask(__name__)
hybrid_agent = None
adk_agent = None
local_llm_with_tools = None
_model_downloading = False
conversation_state = {}
embedding_backfill_attempted = False
_model_init_lock = threading.Lock()

def _load_local_instances():
    """Load local LLM with tools and hybrid agent sharing a single model in memory"""
    global hybrid_agent, local_llm_with_tools
    
    if local_llm_with_tools is None:
        try:
            from software_bug_assistant.tools.local_tools import get_local_llm_with_tools
            local_llm_with_tools = get_local_llm_with_tools()
            if local_llm_with_tools and local_llm_with_tools.is_available():
                print("✓ Local LLM with Tool Calling: READY")
            else:
                local_llm_with_tools = None
        except Exception as e:
            print(f"⚠ Local LLM with tools not available: {e}")
            local_llm_with_tools = None

    try:
        from software_bug_assistant.hybrid_agent import HybridAgent, LocalGemmaModel
        shared_model = local_llm_with_tools.model if local_llm_with_tools else None
        if hybrid_agent is None:
            local_gemma = LocalGemmaModel(model=shared_model)
            hybrid_agent = HybridAgent(enable_local=True, local_model=local_gemma)
        elif hybrid_agent.local_model is None or not hybrid_agent.local_model.is_available():
            hybrid_agent.local_model = LocalGemmaModel(model=shared_model)
            
        if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
            print("✓ Local Gemma-2B: READY")
    except Exception as e:
        print(f"⚠ Local Gemma model init error: {e}")


def init_local_models():
    """Attempt to load Local LLM and Local LLM with tools, sharing model instance and downloading in background if needed"""
    global hybrid_agent, local_llm_with_tools, _model_downloading
    
    # Fast path if already loaded
    if local_llm_with_tools is not None and hybrid_agent is not None and hybrid_agent.local_model is not None and hybrid_agent.local_model.is_available():
        return

    with _model_init_lock:
        model_path = Path("models") / "gemma-2-2b-it-Q4_K_M.gguf"
        
        if not model_path.exists():
            if _model_downloading:
                return
            _model_downloading = True
            def _download_task():
                global _model_downloading, hybrid_agent, local_llm_with_tools
                print("⏳ Gemma model not found. Starting background download (~1.5GB)...")
                try:
                    from download_gemma import download_gemma_model
                    download_gemma_model()
                except Exception as dl_err:
                    print(f"⚠ Background Gemma download failed: {dl_err}")
                    _model_downloading = False
                    return
                _model_downloading = False
                
                if model_path.exists():
                    _load_local_instances()
                            
            threading.Thread(target=_download_task, daemon=True).start()
            return

        # Model file already exists on disk (pre-downloaded in Docker image)
        _load_local_instances()

# Kick off model initialization
init_local_models()

# Initialize direct database access functions (replaces MCP Toolbox)
from software_bug_assistant.tools.db_tools import (
    get_tickets_by_status as db_get_tickets_by_status,
    get_tickets_by_priority as db_get_tickets_by_priority,
)

# Startup DB connection test
try:
    _db_test = db_get_tickets_by_status("Open")
    if _db_test.get('success'):
        print(f"✓ Database connection: OK ({_db_test.get('count', 0)} open tickets)")
    else:
        print(f"⚠ Database connection FAILED: {_db_test.get('error', 'unknown')}")
except Exception as _e:
    print(f"⚠ Database connection test error: {_e}")

def query_database_directly(query: str) -> Optional[str]:
    """
    Directly query the database using psycopg when cloud is unavailable.
    Returns formatted results or None if can't handle.
    """
    query_lower = query.lower()
    
    try:
        result = None
        
        # Detect query type and call appropriate function
        if "open" in query_lower and "ticket" in query_lower:
            result = db_get_tickets_by_status("Open")
        elif "closed" in query_lower and "ticket" in query_lower:
            result = db_get_tickets_by_status("Closed")
        elif "in progress" in query_lower or "in-progress" in query_lower:
            result = db_get_tickets_by_status("In Progress")
        elif "all ticket" in query_lower or "show ticket" in query_lower or "list ticket" in query_lower:
            result = db_get_tickets_by_status("Open")
        elif "high" in query_lower and ("priority" in query_lower or "ticket" in query_lower):
            result = db_get_tickets_by_priority("P1 - High")
        elif "critical" in query_lower:
            result = db_get_tickets_by_priority("P0 - Critical")
        elif "medium" in query_lower and "priority" in query_lower:
            result = db_get_tickets_by_priority("P2 - Medium")
        elif "low" in query_lower and "priority" in query_lower:
            result = db_get_tickets_by_priority("P3 - Low")
        else:
            return None
        
        if result is None or result.get('error'):
            error_msg = result.get('error', 'Unknown error') if result else 'No result'
            return f"⚠️ Database error: {error_msg}\n\nPlease check the database connection and try again."
        
        tickets = result.get('tickets', [])
        
        if isinstance(tickets, list) and len(tickets) > 0:
            # Select key columns for display
            display_cols = ['ticket_id', 'title', 'status', 'priority', 'assignee']
            
            # Build markdown table
            md_table = "| " + " | ".join(display_cols) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(display_cols)) + " |\n"
            
            for row in tickets:
                if isinstance(row, dict):
                    values = []
                    for col in display_cols:
                        val = str(row.get(col, ""))
                        if "@" in val:
                            val = val.split("@")[0]
                        if len(val) > 40:
                            val = val[:37] + "..."
                        values.append(val)
                    md_table += "| " + " | ".join(values) + " |\n"
            
            output = f"📋 **Database Results:**\n\n{md_table}\n\n_Found {len(tickets)} ticket(s)._"
            if result.get('is_fallback'):
                output += (
                    "\n\n---\n> 💡 **Notice:** Remote Supabase database is unreachable (`FATAL: tenant/user not found` — project may be paused).\n"
                    "> Loaded tickets from local database fallback. To connect live Supabase, resume your project in your [Supabase Dashboard](https://supabase.com/dashboard)."
                )
            return output
        elif isinstance(tickets, list) and len(tickets) == 0:
            return "No tickets found matching your query."
        else:
            return f"📋 **Database Results:**\n\n{result}"
            
    except Exception as e:
        print(f"Direct database query failed: {e}")
        import traceback
        traceback.print_exc()
        return None





def _get_client_state(client_id: str) -> dict:
    if client_id not in conversation_state:
        conversation_state[client_id] = {
            "stage": "idle",
            "last_issue": "",
        }
    return conversation_state[client_id]


def _is_affirmative(text: str) -> bool:
    return bool(re.search(r"\b(yes|yep|yeah|fixed|solved|done|see|ok|okay)\b", text))


def _is_negative(text: str) -> bool:
    return bool(re.search(r"\b(no|not yet|still|didn't|did not|failed|nope)\b", text))


def _is_ticket_query(text: str) -> bool:
    has_ticket = bool(re.search(r"\b(ticket|tickets|bug|bugs|issue|issues)\b", text))
    has_action = bool(re.search(r"\b(show|list|get|find|search|open|closed|status|priority|create|update)\b", text))
    return has_ticket and has_action


def _needs_problem_details(text: str) -> bool:
    is_help_request = bool(re.search(r"\b(suggestion|suggest|solution|fix|help|troubleshoot)\b", text))
    has_issue_signal = bool(re.search(r"\b(error|issue|problem|slow|crash|freeze|not working|fail|broken|bug)\b", text))
    return is_help_request and not has_issue_signal


def _is_troubleshoot_request(text: str) -> bool:
    return bool(re.search(r"\b(error|issue|problem|slow|crash|freeze|not working|fail|broken|bug|fix|help|troubleshoot)\b", text))


def _looks_like_new_query(text: str) -> bool:
    """Detect whether user typed a new request instead of yes/no confirmation."""
    if _is_ticket_query(text):
        return True
    if _needs_problem_details(text):
        return True
    if _is_troubleshoot_request(text):
        return True
    # Longer free-text messages are likely new queries, not confirmations.
    if len(text.split()) >= 3 and not _is_affirmative(text) and not _is_negative(text):
        return True
    return False


def _is_generic_issue_text(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return True
    return bool(re.match(r"^(how to fix (this|the) issue\??|fix (this|the) issue\??|this issue\??|issue\??|problem\??|help\??)$", cleaned))


def _extract_issue_from_assistant_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    cleaned = re.split(r"\n\s*---", cleaned, maxsplit=1)[0]
    cleaned = re.split(r"Did that fix it\?", cleaned, flags=re.IGNORECASE, maxsplit=1)[0]
    cleaned = re.split(r"StackOverflow Related Issues", cleaned, flags=re.IGNORECASE, maxsplit=1)[0]
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)

    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s*[-*#>\d.\)]+\s*", "", line).strip()
        if not line:
            continue
        lines.append(line)

    if not lines:
        return ""

    summary = " ".join(lines)
    summary = re.sub(r"\s+", " ", summary).strip()
    return summary[:260]


def _derive_issue_text_for_ticket(last_issue: str, history: list) -> str:
    candidate = (last_issue or "").strip()
    if candidate and not _is_generic_issue_text(candidate):
        return candidate

    for msg in reversed(history or []):
        if msg.get("role") != "assistant":
            continue
        extracted = _extract_issue_from_assistant_text(str(msg.get("content") or ""))
        if extracted and not _is_generic_issue_text(extracted):
            return extracted

    return candidate


def _format_search_section(title: str, items: list, max_items: int = 5) -> str:
    if not items:
        return ""
    lines = [f"## {title}"]
    for i, item in enumerate(items[:max_items], 1):
        name = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet") or item.get("score") or ""
        suffix = f" - {snippet}" if snippet else ""
        lines.append(f"{i}. [{name}]({link}){suffix}")
    return "\n".join(lines)

# HTML Template
HELP_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tandion Bug Assistant | Help & Examples</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --accent: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --dark-light: #1e293b;
            --dark-lighter: #334155;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #475569;
            --glass: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--dark);
            color: var(--text);
            min-height: 100vh;
        }

        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background:
                radial-gradient(ellipse at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 40% 40%, rgba(6, 182, 212, 0.1) 0%, transparent 40%),
                var(--dark);
        }

        .grid-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 50px 50px;
        }

        .page {
            max-width: 1100px;
            margin: 0 auto;
            padding: 32px 24px 64px;
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }

        .title {
            font-size: 28px;
            font-weight: 700;
        }

        .subtitle {
            color: var(--text-muted);
            margin-top: 6px;
            font-size: 14px;
        }

        .back-btn {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            color: var(--text);
            padding: 10px 16px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .back-btn:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--primary);
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
        }

        .card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 18px 20px;
        }

        .card h3 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }

        .card ul {
            padding-left: 16px;
            line-height: 1.6;
        }

        .tip {
            margin-top: 18px;
            padding: 14px 16px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            border: 1px solid var(--glass-border);
            font-size: 13px;
            color: var(--text-muted);
        }

        .section-header {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-muted);
            margin: 28px 0 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .model-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }

        .model-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            padding: 22px 24px;
            position: relative;
            transition: all 0.3s ease;
        }

        .model-card:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
        }

        .model-card.local {
            border-left: 4px solid var(--success);
        }

        .model-card.cloud {
            border-left: 4px solid var(--accent);
        }

        .model-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .model-tag.local {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .model-tag.cloud {
            background: rgba(6, 182, 212, 0.2);
            color: #22d3ee;
            border: 1px solid rgba(6, 182, 212, 0.3);
        }

        .model-card h3 {
            font-size: 18px;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 8px;
            text-transform: none;
            letter-spacing: 0;
        }

        .model-card p {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
            margin-bottom: 16px;
        }

        .syntax-box {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 14px;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .syntax-box code {
            color: #a5b4fc;
            font-weight: 700;
            background: rgba(99, 102, 241, 0.25);
            padding: 3px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .cmd-list {
            list-style: none;
            padding: 0;
            margin: 0;
            font-size: 13px;
        }

        .cmd-list li {
            padding: 6px 0;
            color: var(--text);
            display: flex;
            align-items: baseline;
            gap: 8px;
        }

        .cmd-list li code {
            color: #e2e8f0;
            background: rgba(255, 255, 255, 0.05);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .hybrid-banner {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 18px 22px;
            margin-bottom: 24px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }

        .hybrid-icon {
            font-size: 26px;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .hybrid-text h4 {
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 4px;
            color: #c7d2fe;
        }

        .hybrid-text p {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
            margin: 0;
        }

        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            .model-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    <div class="grid-overlay"></div>

    <div class="page">
        <div class="header">
            <div>
                <div class="title">Help & Examples</div>
                <div class="subtitle">Complete guide to using Local (Offline) & Online (Cloud) AI models, plus ticket management commands.</div>
            </div>
            <a class="back-btn" href="/">← Back to Chat</a>
        </div>

        <div class="section-header">🤖 AI Models: How to Use Local vs Online</div>

        <div class="model-grid">
            <div class="model-card local">
                <span class="model-tag local">🏠 Local • Offline • Gemma-2B</span>
                <h3>Local / Offline Model</h3>
                <p>Runs 100% locally inside the container via llama.cpp CPU inference. Zero external API keys required, private, fast, and operates completely offline.</p>
                <div class="syntax-box">
                    <span>Force Local Prefix:</span>
                    <code>@local &lt;query&gt;</code>
                </div>
                <ul class="cmd-list">
                    <li><code>@local how to debug memory leaks in Python</code></li>
                    <li><code>@local explain deadlock vs race condition</code></li>
                    <li><code>@local what causes segmentation faults in C++?</code></li>
                    <li><code>@local show all open tickets</code></li>
                </ul>
            </div>

            <div class="model-card cloud">
                <span class="model-tag cloud">☁️ Cloud • Online • Gemini</span>
                <h3>Online / Cloud Model</h3>
                <p>Powered by Google Gemini with live tool execution, Google Search, StackOverflow issue cross-referencing, multi-modal screenshot analysis, and deep reasoning.</p>
                <div class="syntax-box">
                    <span>Force Cloud Prefix:</span>
                    <code>@cloud &lt;query&gt;</code>
                </div>
                <ul class="cmd-list">
                    <li><code>@cloud investigate CVE-2024-3094 vulnerability</code></li>
                    <li><code>@cloud analyze this stack trace and suggest root causes</code></li>
                    <li><code>@cloud find solutions for AWS RDS pool exhaustion</code></li>
                    <li>Attach error screenshots using the 📷 camera button</li>
                </ul>
            </div>
        </div>

        <div class="hybrid-banner">
            <div class="hybrid-icon">⚡</div>
            <div class="hybrid-text">
                <h4>Smart Automatic Routing (Default)</h4>
                <p>
                    If you don't prefix with <code>@local</code> or <code>@cloud</code>, Tandion automatically routes simple questions, greetings, and ticket queries to the fast Local model, and complex bug troubleshooting to Cloud Gemini. If external cloud services are ever unavailable, the system automatically falls back to your local model!
                </p>
            </div>
        </div>

        <div class="section-header">📋 Ticket Management Commands</div>

        <div class="grid">
            <div class="card">
                <h3>Create Ticket</h3>
                <ul>
                    <li>Create a ticket: login fails after update</li>
                    <li>Report a bug: dashboard loads forever</li>
                    <li>Open a ticket for VPN not connecting</li>
                </ul>
            </div>

            <div class="card">
                <h3>Update Status</h3>
                <ul>
                    <li>Close ticket 12</li>
                    <li>Resolve ticket #7</li>
                    <li>Mark ticket 5 as in progress</li>
                </ul>
            </div>

            <div class="card">
                <h3>Update Priority</h3>
                <ul>
                    <li>Set ticket 9 to critical</li>
                    <li>Change priority of ticket 3 to P1</li>
                    <li>Make ticket 14 low priority</li>
                </ul>
            </div>

            <div class="card">
                <h3>View & Search</h3>
                <ul>
                    <li>Show all open tickets</li>
                    <li>List critical tickets</li>
                    <li>Search tickets for "login error"</li>
                </ul>
            </div>

            <div class="card">
                <h3>Get Suggestions</h3>
                <ul>
                    <li>Need suggestion</li>
                    <li>I need help</li>
                    <li>Fix my issue</li>
                </ul>
                <div class="tip">After you describe the issue, the assistant will analyze it, show web + StackOverflow results, and ask if it is fixed.</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tandion Bug Assistant | AI-Powered Issue Management</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --dark-bg: #070a13;
            --dark-surface: #0f172a;
            --dark-card: #141f36;
            --dark-border: rgba(255, 255, 255, 0.08);
            --dark-border-subtle: rgba(255, 255, 255, 0.04);
            
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --secondary: #8b5cf6;
            --accent: #06b6d4;
            --accent-glow: rgba(6, 182, 212, 0.25);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            
            --glass-bg: rgba(15, 23, 42, 0.72);
            --glass-card: rgba(20, 31, 54, 0.65);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-hover: rgba(255, 255, 255, 0.06);
            
            --sidebar-width: 280px;
            --header-height: 64px;
            --transition-smooth: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--dark-bg);
            color: var(--text-main);
            height: 100vh;
            overflow: hidden;
            letter-spacing: -0.01em;
        }
        
        /* Modern Ambient Background Mesh */
        .bg-mesh {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -2;
            background: 
                radial-gradient(circle at 10% 90%, rgba(99, 102, 241, 0.18) 0%, transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(139, 92, 246, 0.16) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.08) 0%, transparent 55%),
                var(--dark-bg);
        }
        
        .grid-mesh {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: radial-gradient(circle at 50% 50%, black 50%, transparent 95%);
            -webkit-mask-image: radial-gradient(circle at 50% 50%, black 50%, transparent 95%);
        }
        
        /* Layout Structure */
        .app-container {
            display: flex;
            height: 100vh;
            width: 100%;
            position: relative;
            overflow: hidden;
        }
        
        /* Sidebar Navigation */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--glass-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            padding: 20px 16px;
            transition: var(--transition-smooth);
            z-index: 40;
            flex-shrink: 0;
        }
        
        .app-container.sidebar-collapsed .sidebar {
            width: 0;
            padding: 0;
            opacity: 0;
            border-right: none;
            overflow: hidden;
            pointer-events: none;
        }
        
        .sidebar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 22px;
            padding: 0 4px;
        }
        
        .logo-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: inherit;
        }
        
        .logo-badge {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 0 16px var(--primary-glow);
            flex-shrink: 0;
            transition: transform 0.2s ease;
        }
        
        .logo-wrap:hover .logo-badge {
            transform: scale(1.05);
        }
        
        .logo-text {
            font-size: 16px;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        
        .logo-sub {
            font-size: 11px;
            font-weight: 500;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        
        /* Section Dividers */
        .sidebar-section {
            margin-bottom: 20px;
        }
        
        .section-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-dim);
            margin-bottom: 10px;
            padding: 0 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        /* Status Cards */
        .status-box {
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            transition: var(--transition-smooth);
        }
        
        .status-box:hover {
            border-color: rgba(255, 255, 255, 0.14);
            background: rgba(25, 38, 66, 0.7);
        }
        
        .status-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .status-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
        }
        
        .status-dot-ring {
            position: relative;
            width: 8px;
            height: 8px;
            display: inline-block;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
            display: block;
            transition: var(--transition-smooth);
        }
        
        .status-dot.loading {
            background: var(--warning);
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.6);
            animation: pulse-dot 1.5s infinite;
        }
        
        .status-dot.offline {
            background: var(--danger);
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
        }
        
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(1.2); }
        }
        
        .pill-badge {
            font-size: 10px;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        
        .pill-badge.local {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }
        
        .pill-badge.cloud {
            background: rgba(6, 182, 212, 0.15);
            color: #22d3ee;
            border: 1px solid rgba(6, 182, 212, 0.25);
        }
        
        .status-desc {
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.3;
        }
        
        /* Quick Action List */
        .action-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .action-item-btn {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: var(--transition-smooth);
            text-decoration: none;
            width: 100%;
        }
        
        .action-item-btn:hover {
            background: var(--glass-hover);
            color: var(--text-main);
            border-color: var(--glass-border);
            transform: translateX(3px);
        }
        
        .action-item-btn.help-btn {
            background: rgba(99, 102, 241, 0.12);
            color: #c7d2fe;
            border-color: rgba(99, 102, 241, 0.25);
            font-weight: 600;
        }
        
        .action-item-btn.help-btn:hover {
            background: rgba(99, 102, 241, 0.22);
            color: #ffffff;
            border-color: rgba(99, 102, 241, 0.45);
        }
        
        .action-icon {
            font-size: 15px;
            flex-shrink: 0;
            width: 20px;
            text-align: center;
        }
        
        /* Stats Ribbon */
        .sidebar-bottom {
            margin-top: auto;
            padding-top: 14px;
            border-top: 1px solid var(--dark-border-subtle);
        }
        
        .stats-ribbon {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        
        .stat-card {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--dark-border-subtle);
            border-radius: 10px;
            padding: 8px 10px;
            text-align: center;
        }
        
        .stat-num {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .stat-sub {
            font-size: 10px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }
        
        /* Main Workspace Container */
        .workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            min-width: 0;
            position: relative;
            background: transparent;
        }
        
        /* Header / Navigation Bar */
        .top-navbar {
            height: var(--header-height);
            padding: 0 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--glass-border);
            background: rgba(7, 10, 19, 0.65);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            z-index: 30;
            flex-shrink: 0;
        }
        
        .navbar-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .toggle-btn {
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            color: var(--text-muted);
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            transition: var(--transition-smooth);
        }
        
        .toggle-btn:hover {
            color: var(--text-main);
            background: var(--glass-hover);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        /* OpenWebUI / LibreChat Style Segmented Model Switcher */
        .mode-segmented-control {
            display: flex;
            align-items: center;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--glass-border);
            padding: 3px;
            border-radius: 12px;
            gap: 2px;
        }
        
        .mode-pill-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-family: inherit;
            font-size: 12px;
            font-weight: 500;
            padding: 5px 12px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: var(--transition-smooth);
            white-space: nowrap;
        }
        
        .mode-pill-btn:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.04);
        }
        
        .mode-pill-btn.active {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%);
            color: #ffffff;
            font-weight: 600;
            box-shadow: 0 2px 8px var(--primary-glow);
        }
        
        .mode-pill-btn.active.mode-local {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.85) 0%, rgba(5, 150, 105, 0.85) 100%);
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }
        
        .mode-pill-btn.active.mode-cloud {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.85) 0%, rgba(2, 132, 199, 0.85) 100%);
            box-shadow: 0 2px 8px rgba(6, 182, 212, 0.3);
        }
        
        .navbar-right {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .nav-tool-btn {
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            text-decoration: none;
            transition: var(--transition-smooth);
        }
        
        .nav-tool-btn:hover {
            color: var(--text-main);
            background: var(--glass-hover);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        /* Chat Conversation Feed */
        .chat-feed {
            flex: 1;
            overflow-y: auto;
            padding: 24px 24px 120px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            scroll-behavior: smooth;
        }
        
        .chat-feed::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-feed::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-feed::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        .chat-feed::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Modern Welcome Hero Card */
        .welcome-hero {
            max-width: 820px;
            margin: auto;
            padding: 40px 24px 20px;
            text-align: center;
            animation: fadeIn 0.4s ease forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .hero-avatar-glow {
            width: 68px;
            height: 68px;
            margin: 0 auto 20px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 22px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            box-shadow: 0 0 32px var(--primary-glow);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        
        .hero-title {
            font-size: 30px;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .hero-subtitle {
            font-size: 14px;
            color: var(--text-muted);
            max-width: 580px;
            margin: 0 auto 32px;
            line-height: 1.6;
        }
        
        /* Interactive Prompt Cards Grid */
        .hero-cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 14px;
            text-align: left;
            margin-top: 10px;
        }
        
        .prompt-card {
            background: var(--glass-card);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 16px;
            cursor: pointer;
            transition: var(--transition-smooth);
            position: relative;
            overflow: hidden;
        }
        
        .prompt-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
            opacity: 0;
            transition: opacity 0.25s ease;
        }
        
        .prompt-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
            background: rgba(22, 34, 60, 0.8);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }
        
        .prompt-card:hover::before {
            opacity: 1;
        }
        
        .prompt-card-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }
        
        .prompt-card-icon {
            font-size: 18px;
        }
        
        .prompt-card-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-main);
        }
        
        .prompt-card-desc {
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.45;
            position: relative;
            z-index: 1;
        }
        
        .prompt-card-tag {
            display: inline-block;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            margin-top: 10px;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.3);
            color: #a5b4fc;
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
            z-index: 1;
        }
        
        /* Message Rows */
        .message-row {
            display: flex;
            gap: 14px;
            max-width: 860px;
            width: 100%;
            margin: 0 auto;
            animation: messageSlideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        
        @keyframes messageSlideIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-row.user {
            justify-content: flex-end;
        }
        
        .avatar-wrap {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .message-row.assistant .avatar-wrap {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: #ffffff;
            box-shadow: 0 0 12px var(--primary-glow);
        }
        
        .message-row.user .avatar-wrap {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--glass-border);
            order: 2;
        }
        
        .bubble-wrapper {
            max-width: calc(100% - 50px);
            display: flex;
            flex-direction: column;
        }
        
        .message-row.user .bubble-wrapper {
            align-items: flex-end;
            order: 1;
        }
        
        /* User & Assistant Bubble Styling */
        .msg-bubble {
            border-radius: 16px;
            padding: 14px 18px;
            font-size: 14px;
            line-height: 1.6;
            word-break: break-word;
            position: relative;
        }
        
        .message-row.user .msg-bubble {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%);
            color: #ffffff;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        
        .message-row.assistant .msg-bubble {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-bottom-left-radius: 4px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            color: var(--text-main);
        }
        
        /* Message Footer Toolbar */
        .msg-meta-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 6px;
            padding: 0 4px;
            font-size: 11px;
            color: var(--text-dim);
        }
        
        .model-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 10.5px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        
        .model-chip.local {
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.22);
        }
        
        .model-chip.cloud {
            background: rgba(6, 182, 212, 0.12);
            color: #22d3ee;
            border: 1px solid rgba(6, 182, 212, 0.22);
        }
        
        .copy-msg-btn {
            background: transparent;
            border: none;
            color: var(--text-dim);
            cursor: pointer;
            font-size: 11px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 6px;
            border-radius: 4px;
            transition: var(--transition-smooth);
        }
        
        .copy-msg-btn:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.08);
        }
        
        /* Markdown & Code Block Styling */
        .msg-bubble p {
            margin-bottom: 12px;
        }
        
        .msg-bubble p:last-child {
            margin-bottom: 0;
        }
        
        .msg-bubble ul, .msg-bubble ol {
            margin: 8px 0 12px 20px;
        }
        
        .msg-bubble li {
            margin-bottom: 4px;
        }
        
        .msg-bubble blockquote {
            border-left: 3px solid var(--primary);
            padding: 6px 14px;
            margin: 10px 0;
            background: rgba(99, 102, 241, 0.08);
            border-radius: 0 8px 8px 0;
            color: #cbd5e1;
        }
        
        .msg-bubble code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12.5px;
            background: rgba(0, 0, 0, 0.35);
            padding: 2px 6px;
            border-radius: 6px;
            color: #93c5fd;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        
        /* Advanced Code Block Container */
        .code-block-wrapper {
            margin: 12px 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--glass-border);
            background: rgba(5, 8, 16, 0.7);
        }
        
        .code-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 7px 14px;
            background: rgba(15, 23, 42, 0.9);
            border-bottom: 1px solid var(--glass-border);
            font-size: 11px;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
        }
        
        .copy-code-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 11px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: var(--transition-smooth);
        }
        
        .copy-code-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-main);
        }
        
        .msg-bubble pre {
            margin: 0;
            padding: 14px;
            overflow-x: auto;
            background: transparent;
        }
        
        .msg-bubble pre code {
            background: transparent;
            padding: 0;
            border: none;
            color: #e2e8f0;
            font-size: 12.5px;
            line-height: 1.5;
        }
        
        /* Table Styling */
        .table-container {
            overflow-x: auto;
            margin: 12px 0;
            border-radius: 10px;
            border: 1px solid var(--glass-border);
        }
        
        .msg-bubble table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
        }
        
        .msg-bubble th {
            background: rgba(15, 23, 42, 0.9);
            color: var(--text-main);
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--glass-border);
            font-weight: 600;
        }
        
        .msg-bubble td {
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-muted);
        }
        
        .msg-bubble tr:last-child td {
            border-bottom: none;
        }
        
        .msg-bubble tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
        
        /* Typing Dots Indicator */
        .typing-bubble {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 12px 16px;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            border-bottom-left-radius: 4px;
        }
        
        .typing-dot {
            width: 7px;
            height: 7px;
            background: var(--primary);
            border-radius: 50%;
            animation: typingBounce 1.4s infinite ease-in-out both;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typingBounce {
            0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        .typing-cursor {
            display: inline-block;
            width: 7px;
            height: 15px;
            background: var(--accent);
            margin-left: 4px;
            vertical-align: text-bottom;
            animation: cursorBlink 0.9s infinite;
        }
        
        @keyframes cursorBlink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        
        /* Floating Command Deck / Input Box */
        .command-deck {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0 24px 20px;
            background: linear-gradient(to top, var(--dark-bg) 60%, rgba(7, 10, 19, 0.8) 85%, transparent 100%);
            pointer-events: none;
            z-index: 25;
        }
        
        .command-deck-inner {
            max-width: 860px;
            margin: 0 auto;
            pointer-events: auto;
        }
        
        /* Mode Quick Selector Bar right above input */
        .deck-mode-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 0 4px;
        }
        
        .deck-mode-chips {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .deck-chip {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--glass-border);
            color: var(--text-dim);
            font-size: 11px;
            font-weight: 500;
            padding: 3px 9px;
            border-radius: 20px;
            cursor: pointer;
            transition: var(--transition-smooth);
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .deck-chip:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        .deck-chip.active {
            color: #ffffff;
            background: rgba(99, 102, 241, 0.3);
            border-color: var(--primary);
            font-weight: 600;
        }
        
        .input-box-wrapper {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            padding: 6px 8px 6px 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
            transition: var(--transition-smooth);
        }
        
        .input-box-wrapper:focus-within {
            border-color: rgba(99, 102, 241, 0.6);
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2), 0 0 0 2px rgba(99, 102, 241, 0.15);
        }
        
        .composer-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-main);
            font-size: 14px;
            font-family: inherit;
            padding: 8px 0;
        }
        
        .composer-input::placeholder {
            color: var(--text-dim);
        }
        
        .action-button {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: var(--transition-smooth);
            flex-shrink: 0;
        }
        
        .btn-attach {
            background: var(--glass-hover);
            color: var(--text-muted);
            border: 1px solid var(--glass-border);
        }
        
        .btn-attach:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: scale(1.04);
        }
        
        .btn-send {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%);
            color: #ffffff;
            font-weight: 600;
            width: auto;
            padding: 0 16px;
            gap: 6px;
            font-size: 13px;
            box-shadow: 0 2px 10px var(--primary-glow);
        }
        
        .btn-send:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px var(--primary-glow);
        }
        
        .btn-send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-stop {
            display: none;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #fca5a5;
            font-weight: 600;
            width: auto;
            padding: 0 14px;
            gap: 6px;
            font-size: 13px;
        }
        
        .btn-stop.visible {
            display: flex;
        }
        
        .btn-stop:hover {
            background: rgba(239, 68, 68, 0.25);
            color: #ffffff;
        }
        
        .stop-square {
            width: 10px;
            height: 10px;
            background: currentColor;
            border-radius: 2px;
        }
        
        /* Attachment Preview Chip */
        .attach-tray {
            display: none;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            padding: 6px 12px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            max-width: fit-content;
        }
        
        .attach-tray.visible {
            display: flex;
        }
        
        .attach-thumb {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            object-fit: cover;
            border: 1px solid var(--glass-border);
        }
        
        .attach-name {
            font-size: 12px;
            color: var(--text-muted);
            max-width: 220px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .attach-del-btn {
            background: transparent;
            border: none;
            color: var(--text-dim);
            cursor: pointer;
            font-size: 14px;
            line-height: 1;
            padding: 2px;
            border-radius: 4px;
        }
        
        .attach-del-btn:hover {
            color: var(--danger);
        }
        
        .deck-footer-hint {
            text-align: center;
            font-size: 11px;
            color: var(--text-dim);
            margin-top: 6px;
        }
        
        /* Mobile Overlay Backdrop */
        .sidebar-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            z-index: 35;
        }
        
        /* Responsive Breakpoints */
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                height: 100%;
                transform: translateX(-100%);
                box-shadow: 0 0 30px rgba(0, 0, 0, 0.7);
            }
            
            .app-container.mobile-sidebar-open .sidebar {
                transform: translateX(0);
            }
            
            .app-container.mobile-sidebar-open .sidebar-overlay {
                display: block;
            }
            
            .mode-segmented-control {
                display: none; /* Collapses to quick chips on mobile */
            }
            
            .chat-feed {
                padding: 16px 14px 130px;
            }
            
            .command-deck {
                padding: 0 12px 14px;
            }
            
            .hero-cards-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="bg-mesh"></div>
    <div class="grid-mesh"></div>
    <div class="sidebar-overlay" id="sidebarOverlay" onclick="toggleMobileSidebar(false)"></div>
    
    <div class="app-container" id="appContainer">
        <!-- Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <a href="/" class="logo-wrap">
                    <div class="logo-badge">⚡</div>
                    <div>
                        <div class="logo-text">Tandion</div>
                        <div class="logo-sub">Bug Assistant</div>
                    </div>
                </a>
            </div>
            
            <!-- Live Model Status -->
            <div class="sidebar-section">
                <div class="section-label">AI Runtime Engines</div>
                <div class="status-box">
                    <div class="status-row">
                        <span class="status-title">
                            <span class="status-dot-ring"><span class="status-dot" id="localDot"></span></span>
                            Local Engine
                        </span>
                        <span class="pill-badge local" id="localBadge">Gemma-2B</span>
                    </div>
                    <div class="status-desc">Private • 100% Offline CPU • Free</div>
                </div>
                
                <div class="status-box">
                    <div class="status-row">
                        <span class="status-title">
                            <span class="status-dot-ring"><span class="status-dot" id="cloudDot"></span></span>
                            Cloud Engine
                        </span>
                        <span class="pill-badge cloud" id="cloudBadge">Gemini</span>
                    </div>
                    <div class="status-desc">Deep Reasoning • Live Tools • Search</div>
                </div>
            </div>
            
            <!-- Quick Actions -->
            <div class="sidebar-section">
                <div class="section-label">Quick Actions</div>
                <div class="action-list">
                    <a href="/help" class="action-item-btn help-btn">
                        <span class="action-icon">💡</span>
                        <span>Help & Examples</span>
                    </a>
                    <button class="action-item-btn" onclick="quickAction('Show all open tickets')">
                        <span class="action-icon">📋</span>
                        <span>Open Tickets</span>
                    </button>
                    <button class="action-item-btn" onclick="quickAction('Show critical priority tickets')">
                        <span class="action-icon">🚨</span>
                        <span>Critical Issues</span>
                    </button>
                    <button class="action-item-btn" onclick="quickAction('Show high priority tickets')">
                        <span class="action-icon">⚠️</span>
                        <span>High Priority</span>
                    </button>
                    <button class="action-item-btn" onclick="quickAction('Show closed tickets')">
                        <span class="action-icon">✅</span>
                        <span>Closed Tickets</span>
                    </button>
                </div>
            </div>
            
            <!-- Session Metrics -->
            <div class="sidebar-bottom">
                <div class="stats-ribbon">
                    <div class="stat-card">
                        <div class="stat-num" id="queryCount">0</div>
                        <div class="stat-sub">Queries</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num" id="sessionTime">0m</div>
                        <div class="stat-sub">Session</div>
                    </div>
                </div>
            </div>
        </aside>
        
        <!-- Main Workspace -->
        <main class="workspace">
            <!-- Header Bar -->
            <header class="top-navbar">
                <div class="navbar-left">
                    <button class="toggle-btn" id="sidebarToggle" onclick="handleSidebarToggle()" title="Toggle Sidebar">
                        ☰
                    </button>
                    
                    <!-- Segmented Model Switcher -->
                    <div class="mode-segmented-control" id="modeControl">
                        <button class="mode-pill-btn active" id="btnModeAuto" onclick="setModelMode('auto')" title="Automatic Smart Hybrid Routing">
                            ⚡ Auto Hybrid
                        </button>
                        <button class="mode-pill-btn" id="btnModeLocal" onclick="setModelMode('local')" title="Force Local Offline Gemma-2B Model">
                            🏠 Local Gemma
                        </button>
                        <button class="mode-pill-btn" id="btnModeCloud" onclick="setModelMode('cloud')" title="Force Online Gemini Cloud Model">
                            ☁️ Cloud Gemini
                        </button>
                    </div>
                </div>
                
                <div class="navbar-right">
                    <a href="/help" class="nav-tool-btn" title="View commands and guide">
                        <span>💡</span>
                        <span>Guide</span>
                    </a>
                    <button class="nav-tool-btn" onclick="clearChat()" title="Clear conversation history">
                        <span>🗑️</span>
                        <span>Clear</span>
                    </button>
                    <button class="nav-tool-btn" onclick="refreshStatus()" title="Refresh connection status">
                        <span>🔄</span>
                    </button>
                </div>
            </header>
            
            <!-- Chat Feed Area -->
            <div class="chat-feed" id="chatArea">
                <!-- Welcome Card -->
                <div class="welcome-hero" id="welcomeCard">
                    <div class="hero-avatar-glow">⚡</div>
                    <h2 class="hero-title">Tandion Bug Assistant</h2>
                    <p class="hero-subtitle">
                        High-performance AI bug triage, ticket operations, and stack trace debugging.
                        Choose your routing engine or select an interactive prompt below to begin.
                    </p>
                    
                    <div class="hero-cards-grid">
                        <div class="prompt-card" onclick="quickAction('Show critical priority tickets')">
                            <div class="prompt-card-header">
                                <span class="prompt-card-icon">🚨</span>
                                <span class="prompt-card-title">Critical Bug Triage</span>
                            </div>
                            <div class="prompt-card-desc">Find blocking P0/Critical issues requiring immediate fix.</div>
                            <span class="prompt-card-tag">Show critical priority tickets</span>
                        </div>
                        
                        <div class="prompt-card" onclick="quickAction('@local how to debug memory leaks in Python')">
                            <div class="prompt-card-header">
                                <span class="prompt-card-icon">🏠</span>
                                <span class="prompt-card-title">Local Fast Debug</span>
                            </div>
                            <div class="prompt-card-desc">Private offline CPU analysis for Python memory leak diagnosis.</div>
                            <span class="prompt-card-tag">@local debug memory leaks</span>
                        </div>
                        
                        <div class="prompt-card" onclick="quickAction('@cloud investigate CVE-2024-3094 vulnerability')">
                            <div class="prompt-card-header">
                                <span class="prompt-card-icon">☁️</span>
                                <span class="prompt-card-title">Cloud CVE & Security</span>
                            </div>
                            <div class="prompt-card-desc">Online deep research with live web tools and vulnerability insights.</div>
                            <span class="prompt-card-tag">@cloud investigate CVE-2024-3094</span>
                        </div>
                        
                        <div class="prompt-card" onclick="quickAction('Show all open tickets')">
                            <div class="prompt-card-header">
                                <span class="prompt-card-icon">📋</span>
                                <span class="prompt-card-title">Ticket Dashboard</span>
                            </div>
                            <div class="prompt-card-desc">Query active tickets, assignee distribution, and priority overview.</div>
                            <span class="prompt-card-tag">Show all open tickets</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Floating Command Deck -->
            <div class="command-deck">
                <div class="command-deck-inner">
                    <!-- Deck Mode Chips & Attachment Tray -->
                    <div class="deck-mode-bar">
                        <div class="deck-mode-chips">
                            <span style="font-size: 11px; color: var(--text-dim); margin-right: 2px;">Mode:</span>
                            <button class="deck-chip active" id="chipAuto" onclick="setModelMode('auto')">⚡ Auto</button>
                            <button class="deck-chip" id="chipLocal" onclick="setModelMode('local')">🏠 @local</button>
                            <button class="deck-chip" id="chipCloud" onclick="setModelMode('cloud')">☁️ @cloud</button>
                        </div>
                        
                        <!-- Attachment indicator -->
                        <div class="attach-tray" id="attachTray">
                            <img id="attachThumb" class="attach-thumb" alt="Attachment Preview" />
                            <span id="attachName" class="attach-name">Screenshot attached</span>
                            <button type="button" class="attach-del-btn" onclick="clearPendingImage()" title="Remove attachment">✕</button>
                        </div>
                    </div>
                    
                    <!-- Input Box -->
                    <div class="input-box-wrapper">
                        <input 
                            type="text" 
                            class="composer-input" 
                            id="messageInput"
                            placeholder="Ask about tickets, bugs, or code (Auto Hybrid)..."
                            onkeydown="handleKeyPress(event)"
                            autocomplete="off"
                        />
                        <input type="file" id="imageInput" accept="image/*" style="display:none" />
                        
                        <button class="action-button btn-attach" id="attachButton" onclick="document.getElementById('imageInput').click()" title="Attach Error Screenshot or Image">
                            📷
                        </button>
                        
                        <button class="action-button btn-send" id="sendButton" onclick="sendMessage()">
                            <span>Send</span>
                            <span>→</span>
                        </button>
                        
                        <button class="action-button btn-stop" id="stopButton" onclick="stopResponse()">
                            <div class="stop-square"></div>
                            <span>Stop</span>
                        </button>
                    </div>
                    
                    <div class="deck-footer-hint">
                        Press <strong>Enter ↵</strong> to send • <strong>Shift+Enter</strong> for newline • Paste screenshots directly from clipboard
                    </div>
                </div>
            </div>
        </main>
    </div>
    
    <script>
        // Safe storage helpers with in-memory fallback for sandbox/iframe/privacy shields
        const memoryStorage = {};
        function safeGetItem(type, key) {
            try {
                const storage = type === 'local' ? window.localStorage : window.sessionStorage;
                if (storage) return storage.getItem(key);
            } catch (e) {}
            return memoryStorage[key] || null;
        }

        function safeSetItem(type, key, value) {
            try {
                const storage = type === 'local' ? window.localStorage : window.sessionStorage;
                if (storage) {
                    storage.setItem(key, value);
                    return;
                }
            } catch (e) {}
            memoryStorage[key] = String(value);
        }

        function safeRemoveItem(type, key) {
            try {
                const storage = type === 'local' ? window.localStorage : window.sessionStorage;
                if (storage) {
                    storage.removeItem(key);
                    return;
                }
            } catch (e) {}
            delete memoryStorage[key];
        }

        // Resilient markdown renderer with fallback
        function renderMarkdown(text) {
            if (!text) return '';
            if (window.marked && typeof window.marked.parse === 'function') {
                try {
                    window.marked.setOptions({ gfm: true, breaks: true });
                    return window.marked.parse(text);
                } catch (e) {
                    console.warn('Markdown parsing error:', e);
                }
            }
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML.split(String.fromCharCode(10)).join('<br>');
        }

        // Global State Variables
        let conversationHistory = [];
        const chatStateKey = 'tandion_chat_state';
        const clientIdKey = 'tandion_client_id';
        let clientId = safeGetItem('local', clientIdKey);
        if (!clientId) {
            const fallbackId = 'client_' + Math.random().toString(36).slice(2);
            clientId = (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : fallbackId;
            safeSetItem('local', clientIdKey, clientId);
        }
        
        let queryCount = 0;
        let sessionStart = Date.now();
        let currentAbortController = null;
        let isGenerating = false;
        let currentMode = 'auto'; // 'auto', 'local', 'cloud'
        
        let pendingImageBase64 = null;
        let pendingImageMime = null;
        let pendingImageName = null;
        let pendingImagePreviewUrl = null;

        // Model Switcher Function
        function setModelMode(mode) {
            currentMode = mode;
            const input = document.getElementById('messageInput');
            
            // Update Navbar segmented buttons
            const btnAuto = document.getElementById('btnModeAuto');
            const btnLocal = document.getElementById('btnModeLocal');
            const btnCloud = document.getElementById('btnModeCloud');
            if (btnAuto) {
                btnAuto.className = 'mode-pill-btn' + (mode === 'auto' ? ' active' : '');
            }
            if (btnLocal) {
                btnLocal.className = 'mode-pill-btn' + (mode === 'local' ? ' active mode-local' : '');
            }
            if (btnCloud) {
                btnCloud.className = 'mode-pill-btn' + (mode === 'cloud' ? ' active mode-cloud' : '');
            }
            
            // Update Deck Chips
            const chipAuto = document.getElementById('chipAuto');
            const chipLocal = document.getElementById('chipLocal');
            const chipCloud = document.getElementById('chipCloud');
            if (chipAuto) chipAuto.className = 'deck-chip' + (mode === 'auto' ? ' active' : '');
            if (chipLocal) chipLocal.className = 'deck-chip' + (mode === 'local' ? ' active' : '');
            if (chipCloud) chipCloud.className = 'deck-chip' + (mode === 'cloud' ? ' active' : '');
            
            // Update placeholder
            if (input) {
                if (mode === 'local') {
                    input.placeholder = 'Ask offline model (Gemma-2B)...';
                } else if (mode === 'cloud') {
                    input.placeholder = 'Ask cloud model with tools (Gemini)...';
                } else {
                    input.placeholder = 'Ask about tickets, bugs, or code (Auto Hybrid)...';
                }
            }
        }

        // Sidebar Collapse / Mobile Drawer Handlers
        function handleSidebarToggle() {
            const container = document.getElementById('appContainer');
            if (window.innerWidth <= 768) {
                toggleMobileSidebar(!container.classList.contains('mobile-sidebar-open'));
            } else {
                container.classList.toggle('sidebar-collapsed');
                safeSetItem('local', 'tandion_sidebar_collapsed', container.classList.contains('sidebar-collapsed') ? '1' : '0');
            }
        }

        function toggleMobileSidebar(open) {
            const container = document.getElementById('appContainer');
            if (open) {
                container.classList.add('mobile-sidebar-open');
            } else {
                container.classList.remove('mobile-sidebar-open');
            }
        }

        // Restore sidebar preference on desktop
        if (window.innerWidth > 768 && safeGetItem('local', 'tandion_sidebar_collapsed') === '1') {
            document.getElementById('appContainer').classList.add('sidebar-collapsed');
        }

        function isReloadNavigation() {
            try {
                const navEntries = performance.getEntriesByType && performance.getEntriesByType('navigation');
                if (navEntries && navEntries.length > 0) {
                    return navEntries[0].type === 'reload';
                }
                return performance.navigation && performance.navigation.type === 1;
            } catch (e) {
                return false;
            }
        }

        function saveChatState() {
            try {
                const payload = {
                    conversationHistory,
                    queryCount,
                    sessionStart,
                };
                safeSetItem('session', chatStateKey, JSON.stringify(payload));
            } catch (e) {
                console.warn('saveChatState error:', e);
            }
        }

        function loadChatState() {
            try {
                if (isReloadNavigation()) {
                    safeRemoveItem('session', chatStateKey);
                    return;
                }

                const raw = safeGetItem('session', chatStateKey);
                if (!raw) return;

                const state = JSON.parse(raw);
                conversationHistory = Array.isArray(state.conversationHistory) ? state.conversationHistory : [];
                queryCount = Number.isFinite(state.queryCount) ? state.queryCount : 0;
                sessionStart = Number.isFinite(state.sessionStart) ? state.sessionStart : Date.now();

                const countEl = document.getElementById('queryCount');
                if (countEl) countEl.textContent = queryCount;

                const chatArea = document.getElementById('chatArea');
                const welcome = document.getElementById('welcomeCard');
                if (conversationHistory.length > 0 && welcome) {
                    welcome.remove();
                }

                conversationHistory.forEach(msg => {
                    const isUser = msg.role === 'user';
                    addMessage(msg.content || '', isUser, msg.model_used || null, msg.image_url || null);
                });
            } catch (e) {
                safeRemoveItem('session', chatStateKey);
            }
        }
        
        // Session timer updater
        setInterval(() => {
            const mins = Math.floor((Date.now() - sessionStart) / 60000);
            const timeEl = document.getElementById('sessionTime');
            if (timeEl) timeEl.textContent = mins + 'm';
        }, 60000);
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Image Attachment Handling
        function updateAttachIndicator() {
            const tray = document.getElementById('attachTray');
            const thumb = document.getElementById('attachThumb');
            const nameEl = document.getElementById('attachName');
            
            if (pendingImageName && pendingImagePreviewUrl) {
                tray.classList.add('visible');
                thumb.src = pendingImagePreviewUrl;
                nameEl.textContent = pendingImageName;
            } else {
                tray.classList.remove('visible');
                thumb.src = '';
                nameEl.textContent = '';
            }
        }

        function clearPendingImage() {
            pendingImageBase64 = null;
            pendingImageMime = null;
            pendingImageName = null;
            pendingImagePreviewUrl = null;
            document.getElementById('imageInput').value = '';
            updateAttachIndicator();
        }

        function setPendingImageFromFile(file, labelPrefix = 'Attached image') {
            if (!file) return;

            if (file.size > 5 * 1024 * 1024) {
                alert('Please upload an image smaller than 5MB.');
                return;
            }

            const reader = new FileReader();
            reader.onload = () => {
                const result = String(reader.result || '');
                const parts = result.split(',');
                pendingImageBase64 = parts.length > 1 ? parts[1] : null;
                pendingImageMime = file.type || 'image/png';
                pendingImagePreviewUrl = result;
                pendingImageName = `${labelPrefix}: ${file.name || 'image.png'}`;
                updateAttachIndicator();
            };
            reader.readAsDataURL(file);
        }

        function setPendingImageFromDataUrl(dataUrl, label = 'Pasted screenshot') {
            if (!dataUrl || !dataUrl.startsWith('data:image/')) return false;
            const parts = dataUrl.split(',');
            if (parts.length < 2) return false;
            let mime = 'image/png';
            try {
                const header = parts[0].split(':')[1] || '';
                mime = header.split(';')[0] || 'image/png';
            } catch (e) {}
            pendingImageBase64 = parts[1] || null;
            pendingImageMime = mime;
            pendingImagePreviewUrl = dataUrl;
            pendingImageName = label;
            updateAttachIndicator();
            return true;
        }

        function tryExtractImageFromClipboard(event) {
            const clipboard = event.clipboardData;
            if (!clipboard) return false;

            if (clipboard.files && clipboard.files.length > 0) {
                const file = clipboard.files[0];
                if (file && file.type && file.type.startsWith('image/')) {
                    setPendingImageFromFile(file, 'Clipboard image');
                    return true;
                }
            }

            if (clipboard.items) {
                for (const item of clipboard.items) {
                    if (item.type && item.type.startsWith('image/')) {
                        const file = item.getAsFile();
                        if (file) {
                            setPendingImageFromFile(file, 'Clipboard image');
                            return true;
                        }
                    }
                }
            }

            const html = clipboard.getData && clipboard.getData('text/html');
            if (html) {
                const dataUrlMatch = html.match(/src=["'](data:image[^"']+)["']/i);
                if (dataUrlMatch && setPendingImageFromDataUrl(dataUrlMatch[1])) {
                    return true;
                }
            }

            const text = clipboard.getData && clipboard.getData('text/plain');
            if (text && text.startsWith('data:image/')) {
                return setPendingImageFromDataUrl(text);
            }

            return false;
        }
        
        function quickAction(query) {
            document.getElementById('messageInput').value = query;
            if (window.innerWidth <= 768) {
                toggleMobileSidebar(false);
            }
            sendMessage();
        }
        
        function clearChat() {
            const chatArea = document.getElementById('chatArea');
            chatArea.innerHTML = `
                <div class="welcome-hero" id="welcomeCard">
                    <div class="hero-avatar-glow">⚡</div>
                    <h2 class="hero-title">Tandion Bug Assistant</h2>
                    <p class="hero-subtitle">
                        Conversation cleared. Ready for your next query or bug triage request!
                    </p>
                </div>
            `;
            conversationHistory = [];
            queryCount = 0;
            document.getElementById('queryCount').textContent = queryCount;
            sessionStart = Date.now();
            clearPendingImage();
            saveChatState();
        }
        
        function refreshStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    const localDot = document.getElementById('localDot');
                    const localBadge = document.getElementById('localBadge');
                    if (data.local_available) {
                        localDot.className = 'status-dot';
                        if (localBadge) localBadge.textContent = 'Gemma-2B';
                    } else if (data.local_downloading) {
                        localDot.className = 'status-dot loading';
                        if (localBadge) localBadge.textContent = 'Loading...';
                    } else {
                        localDot.className = 'status-dot offline';
                        if (localBadge) localBadge.textContent = 'Offline';
                    }
                    
                    const cloudDot = document.getElementById('cloudDot');
                    if (cloudDot) {
                        cloudDot.className = 'status-dot' + (data.cloud_available ? '' : ' offline');
                    }
                })
                .catch(err => console.log('Status refresh error:', err));
        }

        // Copy Text Helper
        function copyText(str, btn) {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(str).then(() => {
                    const original = btn.innerHTML;
                    btn.innerHTML = '✓ Copied!';
                    setTimeout(() => { btn.innerHTML = original; }, 2000);
                });
            } else {
                const ta = document.createElement('textarea');
                ta.value = str;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                const original = btn.innerHTML;
                btn.innerHTML = '✓ Copied!';
                setTimeout(() => { btn.innerHTML = original; }, 2000);
            }
        }

        // Enhance rendered code blocks with headers and copy buttons
        function formatCodeBlocks(container) {
            const pres = container.querySelectorAll('pre');
            pres.forEach(pre => {
                if (pre.parentNode && pre.parentNode.classList.contains('code-block-wrapper')) return;
                
                const wrapper = document.createElement('div');
                wrapper.className = 'code-block-wrapper';
                
                const code = pre.querySelector('code');
                let lang = 'CODE';
                if (code) {
                    const classes = code.className.split(' ');
                    for (const c of classes) {
                        if (c.startsWith('language-')) {
                            lang = c.replace('language-', '').toUpperCase();
                            break;
                        }
                    }
                }
                
                const header = document.createElement('div');
                header.className = 'code-header';
                header.innerHTML = `
                    <span>${lang}</span>
                    <button class="copy-code-btn" type="button">📋 Copy</button>
                `;
                
                const copyBtn = header.querySelector('.copy-code-btn');
                copyBtn.addEventListener('click', () => {
                    const text = code ? code.innerText : pre.innerText;
                    copyText(text, copyBtn);
                });
                
                pre.parentNode.insertBefore(wrapper, pre);
                wrapper.appendChild(header);
                wrapper.appendChild(pre);
            });
            
            // Format tables in scrollable container
            const tables = container.querySelectorAll('table');
            tables.forEach(table => {
                if (table.parentNode && table.parentNode.classList.contains('table-container')) return;
                const tblWrap = document.createElement('div');
                tblWrap.className = 'table-container';
                table.parentNode.insertBefore(tblWrap, table);
                tblWrap.appendChild(table);
            });
        }
        
        function addMessage(text, isUser, modelUsed = null, imageUrl = null) {
            const chatArea = document.getElementById('chatArea');
            const welcome = document.getElementById('welcomeCard');
            if (welcome) welcome.remove();
            
            const messageRow = document.createElement('div');
            messageRow.className = `message-row ${isUser ? 'user' : 'assistant'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar-wrap';
            avatar.textContent = isUser ? '👤' : '⚡';
            
            const bubbleWrapper = document.createElement('div');
            bubbleWrapper.className = 'bubble-wrapper';
            
            const bubble = document.createElement('div');
            bubble.className = 'msg-bubble';
            
            if (isUser) {
                if (imageUrl) {
                    const img = document.createElement('img');
                    img.src = imageUrl;
                    img.alt = 'Uploaded screenshot';
                    img.style.maxWidth = '300px';
                    img.style.maxHeight = '240px';
                    img.style.borderRadius = '10px';
                    img.style.display = 'block';
                    img.style.marginBottom = text ? '8px' : '0';
                    bubble.appendChild(img);
                }
                if (text) {
                    const txt = document.createElement('div');
                    txt.textContent = text;
                    bubble.appendChild(txt);
                }
            } else {
                bubble.innerHTML = renderMarkdown(text);
                formatCodeBlocks(bubble);
            }
            
            bubbleWrapper.appendChild(bubble);
            
            if (!isUser) {
                const metaBar = document.createElement('div');
                metaBar.className = 'msg-meta-bar';
                
                let modelLabel = 'Gemini';
                let tagClass = 'cloud';
                let modelIcon = '☁️';
                
                if (modelUsed === 'gemma-2b-tools') {
                    modelLabel = 'Gemma-2B + Tools';
                    tagClass = 'local';
                    modelIcon = '🔧';
                } else if (modelUsed === 'gemma-2b-local') {
                    modelLabel = 'Gemma-2B Local';
                    tagClass = 'local';
                    modelIcon = '🏠';
                }
                
                metaBar.innerHTML = `
                    <span class="model-chip ${tagClass}">
                        ${modelIcon} ${modelLabel}
                    </span>
                    <span>${new Date().toLocaleTimeString()}</span>
                    <button class="copy-msg-btn" type="button" title="Copy response">
                        📋 Copy
                    </button>
                `;
                
                const copyBtn = metaBar.querySelector('.copy-msg-btn');
                copyBtn.addEventListener('click', () => {
                    copyText(text, copyBtn);
                });
                
                bubbleWrapper.appendChild(metaBar);
            }
            
            messageRow.appendChild(avatar);
            messageRow.appendChild(bubbleWrapper);
            chatArea.appendChild(messageRow);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendButton');
            const chatArea = document.getElementById('chatArea');
            
            let message = input.value.trim();
            if (!message && !pendingImageBase64) return;

            // Apply active mode prefix automatically if not already present
            if (currentMode === 'local' && message && !message.startsWith('@local') && !message.startsWith('@cloud')) {
                message = '@local ' + message;
            } else if (currentMode === 'cloud' && message && !message.startsWith('@cloud') && !message.startsWith('@local')) {
                message = '@cloud ' + message;
            }

            const imagePayloadBase64 = pendingImageBase64;
            const imagePayloadMime = pendingImageMime;
            const imagePreviewUrl = pendingImagePreviewUrl;

            const userMessageText = message;
            const historyUserText = message || 'Uploaded image';
            
            addMessage(userMessageText, true, null, imagePreviewUrl);
            input.value = '';
            clearPendingImage();
            sendBtn.disabled = true;
            
            // Show stop button
            const stopBtn = document.getElementById('stopButton');
            stopBtn.classList.add('visible');
            isGenerating = true;
            
            // Create abort controller
            currentAbortController = new AbortController();
            
            // Show typing indicator
            const typingRow = document.createElement('div');
            typingRow.className = 'message-row assistant';
            typingRow.id = 'typingRow';
            typingRow.innerHTML = `
                <div class="avatar-wrap">⚡</div>
                <div class="bubble-wrapper">
                    <div class="typing-bubble">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            chatArea.appendChild(typingRow);
            chatArea.scrollTop = chatArea.scrollHeight;
            
            let assistantRow = null;
            let assistantBubble = null;
            let assistantWrapper = null;
            let accumulatedText = "";
            let modelUsed = null;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message,
                        history: conversationHistory,
                        client_id: clientId,
                        image_data: imagePayloadBase64,
                        image_mime: imagePayloadMime,
                        stream: true
                    }),
                    signal: currentAbortController.signal
                });
                
                const contentType = response.headers.get('content-type') || '';
                
                if (contentType.includes('text/event-stream')) {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder('utf-8');
                    let buffer = '';

                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split(String.fromCharCode(10));
                        buffer = lines.pop();

                        for (const line of lines) {
                            const trimmed = line.trim();
                            if (!trimmed.startsWith('data:')) continue;
                            const jsonStr = trimmed.substring(5).trim();
                            if (!jsonStr) continue;

                            try {
                                const event = JSON.parse(jsonStr);

                                if (event.type === 'meta') {
                                    modelUsed = event.model_used;
                                    const tr = document.getElementById('typingRow');
                                    if (tr) tr.remove();

                                    if (!assistantRow) {
                                        assistantRow = document.createElement('div');
                                        assistantRow.className = 'message-row assistant';

                                        const avatar = document.createElement('div');
                                        avatar.className = 'avatar-wrap';
                                        avatar.textContent = '⚡';

                                        assistantWrapper = document.createElement('div');
                                        assistantWrapper.className = 'bubble-wrapper';

                                        assistantBubble = document.createElement('div');
                                        assistantBubble.className = 'msg-bubble';
                                        assistantBubble.innerHTML = '<span class="typing-cursor"></span>';

                                        assistantWrapper.appendChild(assistantBubble);
                                        assistantRow.appendChild(avatar);
                                        assistantRow.appendChild(assistantWrapper);

                                        chatArea.appendChild(assistantRow);
                                        chatArea.scrollTop = chatArea.scrollHeight;
                                    }
                                } else if (event.type === 'chunk') {
                                    const tr = document.getElementById('typingRow');
                                    if (tr) tr.remove();

                                    if (!assistantRow) {
                                        assistantRow = document.createElement('div');
                                        assistantRow.className = 'message-row assistant';

                                        const avatar = document.createElement('div');
                                        avatar.className = 'avatar-wrap';
                                        avatar.textContent = '⚡';

                                        assistantWrapper = document.createElement('div');
                                        assistantWrapper.className = 'bubble-wrapper';

                                        assistantBubble = document.createElement('div');
                                        assistantBubble.className = 'msg-bubble';
                                        assistantWrapper.appendChild(assistantBubble);
                                        assistantRow.appendChild(avatar);
                                        assistantRow.appendChild(assistantWrapper);
                                        chatArea.appendChild(assistantRow);
                                    }

                                    accumulatedText += event.text;
                                    assistantBubble.innerHTML = renderMarkdown(accumulatedText) + '<span class="typing-cursor"></span>';
                                    formatCodeBlocks(assistantBubble);
                                    chatArea.scrollTop = chatArea.scrollHeight;
                                } else if (event.type === 'done') {
                                    if (assistantBubble) {
                                        assistantBubble.innerHTML = renderMarkdown(accumulatedText);
                                        formatCodeBlocks(assistantBubble);

                                        if (assistantWrapper && !assistantWrapper.querySelector('.msg-meta-bar')) {
                                            const metaBar = document.createElement('div');
                                            metaBar.className = 'msg-meta-bar';
                                            
                                            let modelLabel = 'Gemini';
                                            let tagClass = 'cloud';
                                            let modelIcon = '☁️';
                                            
                                            if (modelUsed === 'gemma-2b-tools') {
                                                modelLabel = 'Gemma-2B + Tools';
                                                tagClass = 'local';
                                                modelIcon = '🔧';
                                            } else if (modelUsed === 'gemma-2b-local') {
                                                modelLabel = 'Gemma-2B Local';
                                                tagClass = 'local';
                                                modelIcon = '🏠';
                                            }
                                            
                                            metaBar.innerHTML = `
                                                <span class="model-chip ${tagClass}">
                                                    ${modelIcon} ${modelLabel}
                                                </span>
                                                <span>${new Date().toLocaleTimeString()}</span>
                                                <button class="copy-msg-btn" type="button" title="Copy response">
                                                    📋 Copy
                                                </button>
                                            `;
                                            
                                            const copyBtn = metaBar.querySelector('.copy-msg-btn');
                                            copyBtn.addEventListener('click', () => {
                                                copyText(accumulatedText, copyBtn);
                                            });
                                            
                                            assistantWrapper.appendChild(metaBar);
                                        }
                                    }

                                    conversationHistory.push({
                                        role: 'user',
                                        content: historyUserText,
                                        image_url: imagePreviewUrl,
                                    });
                                    conversationHistory.push({ role: 'assistant', content: accumulatedText, model_used: modelUsed });
                                    queryCount++;
                                    document.getElementById('queryCount').textContent = queryCount;
                                    saveChatState();
                                }
                            } catch (parseErr) {
                                console.warn('Error parsing SSE event:', parseErr);
                            }
                        }
                    }
                } else {
                    const tr = document.getElementById('typingRow');
                    if (tr) tr.remove();

                    if (!response.ok) {
                        let errMsg = `Server returned status ${response.status}`;
                        try {
                            const errData = await response.json();
                            if (errData && errData.error) errMsg = errData.error;
                        } catch (_) {}
                        addMessage('Error: ' + errMsg, false);
                        return;
                    }

                    const data = await response.json();
                    if (data.error) {
                        addMessage('Error: ' + data.error, false);
                    } else {
                        addMessage(data.response, false, data.model_used);
                        conversationHistory.push({
                            role: 'user',
                            content: historyUserText,
                            image_url: imagePreviewUrl,
                        });
                        conversationHistory.push({ role: 'assistant', content: data.response, model_used: data.model_used });
                        
                        queryCount++;
                        document.getElementById('queryCount').textContent = queryCount;
                        saveChatState();
                    }
                }
            } catch (error) {
                const tr = document.getElementById('typingRow');
                if (tr) tr.remove();
                
                if (error.name === 'AbortError') {
                    if (assistantBubble) {
                        assistantBubble.innerHTML = renderMarkdown(accumulatedText) + ' *(stopped)*';
                        formatCodeBlocks(assistantBubble);
                    } else {
                        addMessage('⏹️ Generation stopped by user.', false);
                    }
                } else {
                    addMessage('Connection error. Please try again: ' + error.message, false);
                }
            } finally {
                stopBtn.classList.remove('visible');
                isGenerating = false;
                currentAbortController = null;
                sendBtn.disabled = false;
                input.focus();
            }
        }
        
        function stopResponse() {
            if (currentAbortController && isGenerating) {
                currentAbortController.abort();
                isGenerating = false;
                document.getElementById('stopButton').classList.remove('visible');
            }
        }
        
        // Event Listeners
        const imageInputEl = document.getElementById('imageInput');
        if (imageInputEl) {
            imageInputEl.addEventListener('change', event => {
                const file = event.target.files && event.target.files[0];
                if (!file) return;
                setPendingImageFromFile(file, 'Attached screenshot');
            });
        }

        const msgInputEl = document.getElementById('messageInput');
        if (msgInputEl) {
            msgInputEl.addEventListener('keydown', handleKeyPress);
            msgInputEl.addEventListener('paste', event => {
                if (tryExtractImageFromClipboard(event)) {
                    event.preventDefault();
                }
            });
        }

        document.addEventListener('paste', event => {
            const target = event.target;
            if (target && target.id === 'messageInput') return;
            if (tryExtractImageFromClipboard(event)) {
                event.preventDefault();
            }
        });

        // Window Globals for Event Handlers
        window.sendMessage = sendMessage;
        window.handleKeyPress = handleKeyPress;
        window.quickAction = quickAction;
        window.clearChat = clearChat;
        window.stopResponse = stopResponse;
        window.clearPendingImage = clearPendingImage;
        window.setModelMode = setModelMode;
        window.handleSidebarToggle = handleSidebarToggle;
        window.toggleMobileSidebar = toggleMobileSidebar;
        window.copyText = copyText;

        loadChatState();
        refreshStatus();
        setInterval(refreshStatus, 4000);
    </script>
</body>
</html>

"""

@app.route('/')
def index():
    """Serve the main UI"""
    resp = make_response(render_template_string(HTML_TEMPLATE))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/help')
def help_page():
    """Serve the help/examples page"""
    resp = make_response(render_template_string(HELP_TEMPLATE))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


def init_agents():
    """Initialize both hybrid agent and ADK agent"""
    global hybrid_agent, adk_agent, embedding_backfill_attempted
    
    # Initialize hybrid agent (for local Gemma)
    if hybrid_agent is None:
        try:
            from software_bug_assistant.hybrid_agent import HybridAgent, LocalGemmaModel
            shared_model = local_llm_with_tools.model if local_llm_with_tools else None
            local_gemma = LocalGemmaModel(model=shared_model)
            hybrid_agent = HybridAgent(enable_local=True, local_model=local_gemma)
        except Exception as e:
            print(f"Warning: Hybrid agent init failed: {e}")

    if not embedding_backfill_attempted and os.getenv("AUTO_EMBED_ON_START", "").upper() == "TRUE":
        embedding_backfill_attempted = True
        backfill_missing_embeddings()
    
    # Initialize ADK agent (for cloud Gemini with tools)
    if adk_agent is None:
        try:
            # Configure google-genai client BEFORE importing agent
            # ADK uses google-genai (not google-generativeai)
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                # Create a client to verify the API key works
                client = genai.Client(api_key=api_key)
            
            # Temporarily allow agent import
            original_skip = os.environ.get("SKIP_AGENT_IMPORT")
            os.environ["SKIP_AGENT_IMPORT"] = "FALSE"
            
            from software_bug_assistant.agent import root_agent
            adk_agent = root_agent
            
            # Restore setting
            if original_skip:
                os.environ["SKIP_AGENT_IMPORT"] = original_skip
            else:
                os.environ.pop("SKIP_AGENT_IMPORT", None)
            
            print("✓ ADK agent initialized for cloud Gemini")
        except Exception as e:
            print(f"Warning: ADK agent init failed: {e}")
            adk_agent = None


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get status of local and cloud models"""
    global hybrid_agent, adk_agent, local_llm_with_tools, _model_downloading
    
    init_agents()
    init_local_models()
    
    local_available = (
        (hybrid_agent is not None and hybrid_agent.local_model is not None and hybrid_agent.local_model.is_available()) or
        (local_llm_with_tools is not None and local_llm_with_tools.is_available())
    )
    
    cloud_available = adk_agent is not None
    
    return jsonify({
        "local_available": local_available,
        "local_downloading": _model_downloading,
        "cloud_available": cloud_available
    })


def sse_event(payload: dict) -> str:
    """Format SSE event string for client streaming."""
    return f"data: {json.dumps(payload)}\n\n"


def _get_fast_greeting(msg_lower: str) -> Optional[str]:
    """Return an instant, high-quality greeting/intro (< 20ms) for common pleasantries."""
    cleaned = re.sub(r'[^\w\s]', '', msg_lower).strip()
    greetings = {
        "hi": "Hello! 👋 I'm your Tandion Software Bug Assistant. How can I assist you with your software, tickets, or troubleshooting today?",
        "hello": "Hello! 👋 I'm your Tandion Software Bug Assistant. How can I help you troubleshoot a bug or manage your tickets today?",
        "hey": "Hey there! 👋 What issue or ticket can I help you with today?",
        "good morning": "Good morning! ☀️ How can I assist you with software debugging or tickets today?",
        "good evening": "Good evening! 🌙 How can I help you today?",
        "who are you": "I am the **Hybrid AI Software Bug Assistant** for **Tandion IT Software**. I combine local Gemma-2B processing with Cloud Gemini to help you triage errors, search documentation, manage support tickets, and resolve bugs efficiently.",
        "what can you do": "Here is what I can do for you:\n\n- 🔍 **Triage & Debug Code**: Analyze stack traces, syntax errors, and system exceptions.\n- 🎫 **Manage Tickets**: View open/closed tickets, update status and priority, or create new bug tickets directly.\n- 📚 **Search Solutions**: Find relevant troubleshooting guides and StackOverflow discussions.\n- 🏠 **Hybrid Local & Cloud**: Run queries locally with `@local` (Gemma-2B) or use advanced reasoning with `@cloud` (Gemini).",
        "help": "How can I help you? You can:\n1. Describe any bug or error message (or paste a screenshot).\n2. Ask to view tickets (e.g. *\"Show all open tickets\"* or *\"Ticket #11\"*).\n3. Update tickets (e.g. *\"Close ticket 11\"* or *\"Change priority of ticket 11 to critical\"*).\n4. Use `@local` to run locally on Gemma-2B or `@cloud` for Cloud Gemini."
    }
    return greetings.get(cleaned)


def stream_cloud_gemini(adk_agent_instance, issue_text: str, image_bytes=None, image_mime="image/png"):
    """Stream Gemini Cloud tokens through a thread-safe queue for real-time SSE delivery."""
    q = queue.Queue()

    def worker():
        try:
            import asyncio
            import uuid
            from google.adk.runners import InMemoryRunner
            from google.genai import types

            async def run_async_agent(include_image: bool = True):
                parts = [types.Part(text=issue_text)]
                if include_image and image_bytes is not None:
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
                content = types.Content(role="user", parts=parts)

                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*mismatch.*")
                    runner = InMemoryRunner(agent=adk_agent_instance, app_name="software_bug_assistant")

                user_id = "web_ui_user"
                session_id = f"session_{uuid.uuid4().hex[:8]}"

                await runner.session_service.create_session(
                    app_name="software_bug_assistant",
                    user_id=user_id,
                    session_id=session_id,
                )

                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=content
                ):
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts') and event.content.parts:
                            for response_part in event.content.parts:
                                if hasattr(response_part, 'text') and response_part.text:
                                    q.put(("chunk", response_part.text))
                        elif hasattr(event.content, 'text') and event.content.text:
                            q.put(("chunk", event.content.text))
                    elif hasattr(event, 'text') and event.text:
                        q.put(("chunk", event.text))
                    elif isinstance(event, str) and event:
                        q.put(("chunk", event))

            try:
                asyncio.run(run_async_agent(include_image=True))
            except Exception as image_run_err:
                if image_bytes is not None:
                    print(f"Cloud triage image processing failed, retrying text-only: {image_run_err}")
                    q.put(("chunk", "I couldn't fully process the image, so I used your text prompt instead.\n\n"))
                    asyncio.run(run_async_agent(include_image=False))
                else:
                    raise image_run_err

            # Append StackExchange results non-blockingly at the end of the stream
            try:
                stack = query_stackexchange(issue_text)
                stack_section = _format_search_section("📚 StackOverflow Related Issues", stack.get("results", []))
                if stack_section:
                    q.put(("chunk", "\n\n---\n\n" + stack_section))
            except Exception as search_err:
                print(f"StackExchange search skipped: {search_err}")

            q.put(("done", None))
        except Exception as e:
            q.put(("error", e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        try:
            kind, val = q.get(timeout=100)
            if kind == "chunk":
                yield val
            elif kind == "done":
                break
            elif kind == "error":
                raise val
        except queue.Empty:
            break


def send_chat_response(
    content,
    model_used: str,
    classification: str,
    ask_fix_confirmation: bool = False,
    is_stream: bool = False,
    error: bool = False,
    state: Optional[dict] = None
):
    """Unified response builder supporting both real-time SSE streaming and standard JSON."""
    if not is_stream:
        if isinstance(content, str):
            final_text = content
        else:
            final_text = "".join(list(content))
        if ask_fix_confirmation:
            final_text += "\n\n---\n\nDid that fix it? Reply **yes** or **no**. If no, I will create a ticket."
            if state is not None:
                state["stage"] = "awaiting_fix_confirmation"
        resp_payload = {
            "response": final_text,
            "model_used": model_used,
            "classification": classification
        }
        if error:
            resp_payload["error"] = True
        return jsonify(resp_payload)
    
    # Streaming Response via Server-Sent Events (SSE)
    def event_stream():
        yield sse_event({"type": "meta", "model_used": model_used, "classification": classification})
        if isinstance(content, str):
            yield sse_event({"type": "chunk", "text": content})
        else:
            try:
                for chunk in content:
                    if chunk:
                        yield sse_event({"type": "chunk", "text": chunk})
            except Exception as stream_err:
                print(f"Streaming error: {stream_err}")
                yield sse_event({"type": "chunk", "text": f"\n\n*(Stream error: {stream_err})*"})
        if ask_fix_confirmation:
            yield sse_event({"type": "chunk", "text": "\n\n---\n\nDid that fix it? Reply **yes** or **no**. If no, I will create a ticket."})
            if state is not None:
                state["stage"] = "awaiting_fix_confirmation"
        yield sse_event({"type": "done"})

    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and route to appropriate model with optional live streaming"""
    global hybrid_agent, adk_agent, local_llm_with_tools, _model_downloading
    
    # Initialize agents if needed
    init_agents()
    init_local_models()
    
    if hybrid_agent is None:
        return jsonify({
            "error": "Failed to initialize hybrid agent"
        }), 500
    
    data = request.json or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    client_id = data.get('client_id') or request.remote_addr or "anonymous"
    image_data = data.get('image_data')
    image_mime = data.get('image_mime', 'image/png')
    is_stream = bool(data.get('stream', False))
    state = _get_client_state(client_id)

    image_bytes = None
    if image_data:
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            return jsonify({"error": "Invalid image payload"}), 400
    
    if not message and image_bytes is None:
        return jsonify({"error": "Message or image is required"}), 400

    # Check for @cloud prefix to force Cloud Gemini
    force_cloud = False
    if message.lower().startswith('@cloud'):
        force_cloud = True
        message = re.sub(r'^@cloud\s*:?', '', message, flags=re.IGNORECASE).strip()
        if not message:
            return send_chat_response(
                "Please add a query after @cloud, for example: @cloud show open tickets",
                model_used="gemini-cloud",
                classification="FORCED_CLOUD",
                error=True,
                is_stream=is_stream,
                state=state
            )
        print(f"🌩️ Force Cloud Gemini requested for: {message}")

    # Check for @local prefix to force Local Gemma-2B
    force_local = False
    if message.lower().startswith('@local'):
        force_local = True
        message = re.sub(r'^@local\s*:?', '', message, flags=re.IGNORECASE).strip()
        if not message:
            return send_chat_response(
                "Please add a query after @local, for example: @local how to troubleshoot slow queries",
                model_used="gemma-2b-local",
                classification="FORCED_LOCAL",
                error=True,
                is_stream=is_stream,
                state=state
            )
        print(f"🏠 Force Local Gemma requested for: {message}")

    if image_bytes is not None:
        force_cloud = True
        if not message:
            message = "Analyze the uploaded image and provide troubleshooting steps."

    message_lower = message.lower()

    if force_cloud and state["stage"] == "awaiting_fix_confirmation":
        state["stage"] = "idle"
        state["last_issue"] = ""

    if state["stage"] == "awaiting_fix_confirmation":
        if _is_affirmative(message_lower):
            state["stage"] = "idle"
            state["last_issue"] = ""
            return send_chat_response(
                "✅ Glad it is fixed. I will not create a ticket. Anything else you need?",
                model_used="triage",
                classification="FIX_CONFIRMED",
                is_stream=is_stream,
                state=state
            )
        if _is_negative(message_lower):
            issue_text = _derive_issue_text_for_ticket(state.get("last_issue", ""), history)
            state["stage"] = "idle"
            state["last_issue"] = ""
            if not issue_text:
                return send_chat_response(
                    "I can create a ticket, but I need the issue details first. What problem should I file?",
                    model_used="triage",
                    classification="NEED_DETAILS",
                    is_stream=is_stream,
                    state=state
                )
            ticket_creator = get_ticket_creator()
            ticket_response = ticket_creator.create_ticket_from_query(issue_text)
            return send_chat_response(
                ticket_response,
                model_used="database-direct",
                classification="TICKET_CREATED",
                is_stream=is_stream,
                state=state
            )

        if _looks_like_new_query(message_lower):
            state["stage"] = "idle"
            state["last_issue"] = ""
        else:
            return send_chat_response(
                "Please confirm if the issue is fixed. Reply 'yes' or 'no'.",
                model_used="triage",
                classification="NEED_CONFIRMATION",
                is_stream=is_stream,
                state=state
            )

    if state["stage"] == "awaiting_problem":
        state["stage"] = "idle"
        message_lower = message.lower()
        state["last_issue"] = message
    
    # Build context from conversation history
    context = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversation_history": [
            f"{msg['role']}: {msg['content']}" 
            for msg in history[-5:]
        ] if history else []
    }

    if _needs_problem_details(message_lower):
        state["stage"] = "awaiting_problem"
        return send_chat_response(
            "What problem should I fix? Please describe the issue (symptoms, error text, when it happens).",
            model_used="triage",
            classification="NEED_DETAILS",
            is_stream=is_stream,
            state=state
        )

    # 0. Fast-path for common pleasantries & greetings (< 20ms response time)
    if not force_cloud and not force_local:
        fast_greeting = _get_fast_greeting(message_lower)
        if fast_greeting:
            return send_chat_response(
                fast_greeting,
                model_used="gemma-2b-local",
                classification="FAST_GREETING",
                is_stream=is_stream,
                state=state
            )

    # If user explicitly requested @local, handle exclusively with local model/tools
    if force_local:
        # Check direct database and tools for ticket-related queries
        if _is_ticket_query(message_lower):
            db_result = query_database_directly(message)
            if db_result:
                return send_chat_response(db_result, "gemma-2b-tools", "FORCED_LOCAL", is_stream=is_stream, state=state)
            if local_llm_with_tools and local_llm_with_tools.is_available():
                try:
                    local_tool_response = local_llm_with_tools.chat(message)
                    if local_tool_response and local_tool_response.strip():
                        return send_chat_response(local_tool_response, "gemma-2b-tools", "FORCED_LOCAL", is_stream=is_stream, state=state)
                except Exception as e:
                    print(f"Force local tools error: {e}")

        # For conversational / technical questions with @local, stream directly via Gemma-2B
        if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
            try:
                if is_stream:
                    gen = hybrid_agent.local_model.generate_stream(message, max_tokens=150, context=context)
                    return send_chat_response(gen, "gemma-2b-local", "FORCED_LOCAL", is_stream=True, state=state)
                else:
                    local_response = hybrid_agent.local_model.generate(message, max_tokens=150, context=context)
                    if local_response and local_response.strip():
                        return send_chat_response(local_response, "gemma-2b-local", "FORCED_LOCAL", is_stream=False, state=state)
            except Exception as e:
                print(f"Force local model error: {e}")

        # Fallback to local tools if direct generation was unavailable or empty
        if local_llm_with_tools and local_llm_with_tools.is_available():
            try:
                local_tool_response = local_llm_with_tools.chat(message)
                if local_tool_response and local_tool_response.strip():
                    return send_chat_response(local_tool_response, "gemma-2b-tools", "FORCED_LOCAL", is_stream=is_stream, state=state)
            except Exception as e:
                print(f"Force local tools fallback error: {e}")

        if _model_downloading:
            return send_chat_response(
                "⏳ **Local Gemma-2B is currently downloading in the background (~1.5GB).**\n\nPlease wait a moment for the download to complete, or ask without `@local` to use Cloud Gemini in the meantime.",
                model_used="offline",
                classification="FORCED_LOCAL",
                error=True,
                is_stream=is_stream,
                state=state
            )

        return send_chat_response(
            "⚠️ Local Gemma-2B model is not available right now. Please try again in a few moments, or ask without `@local` to use Cloud Gemini.",
            model_used="offline",
            classification="FORCED_LOCAL",
            error=True,
            is_stream=is_stream,
            state=state
        )

    # 1. Simple queries (Greetings, general assistant questions, small talk) -> Local Gemma-2B
    if not force_cloud and QueryClassifier.is_simple(message):
        if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
            try:
                if is_stream:
                    gen = hybrid_agent.local_model.generate_stream(message, max_tokens=100, context=context)
                    return send_chat_response(gen, "gemma-2b-local", "SIMPLE", is_stream=True, state=state)
                else:
                    local_response = hybrid_agent.local_model.generate(message, max_tokens=100, context=context)
                    if local_response and local_response.strip():
                        return send_chat_response(local_response, "gemma-2b-local", "SIMPLE", is_stream=False, state=state)
            except Exception as e:
                print(f"Local model simple query error: {e}")

    # 2. Ticket queries & direct database lookups -> Local database / Local tools
    if not force_cloud and _is_ticket_query(message_lower):
        db_result = query_database_directly(message)
        if db_result:
            return send_chat_response(db_result, "gemma-2b-tools", "DATABASE_DIRECT", is_stream=is_stream, state=state)
        
        if local_llm_with_tools and local_llm_with_tools.is_available():
            try:
                local_tool_response = local_llm_with_tools.chat(message)
                if local_tool_response and local_tool_response.strip():
                    return send_chat_response(local_tool_response, "gemma-2b-tools", "TICKET_TOOL", is_stream=is_stream, state=state)
            except Exception as e:
                print(f"Local tools error on ticket query: {e}")

    # 3. Technical issues, code debugging, and triage -> Cloud Gemini (with local fallback)
    issue_text = state.get("last_issue") or message
    state["last_issue"] = issue_text
    ask_fix_confirmation = _is_troubleshoot_request(message_lower)

    # If user explicitly requested @cloud, but Cloud Gemini is not configured/available
    if force_cloud and adk_agent is None:
        return send_chat_response(
            "@cloud was requested, but Cloud Gemini is not available right now. Check API key/network and try again.",
            model_used="gemini-cloud",
            classification="FORCED_CLOUD",
            error=True,
            is_stream=is_stream,
            state=state
        )

    # Try Cloud Gemini first when available
    if adk_agent is not None:
        try:
            if is_stream:
                gen = stream_cloud_gemini(adk_agent, issue_text, image_bytes, image_mime)
                return send_chat_response(
                    gen,
                    model_used="gemini-cloud",
                    classification="FORCED_CLOUD" if force_cloud else "TRIAGE",
                    ask_fix_confirmation=ask_fix_confirmation,
                    is_stream=True,
                    state=state
                )
            else:
                import asyncio
                import uuid
                from google.adk.runners import InMemoryRunner
                from google.genai import types

                async def run_agent_async(include_image: bool = True):
                    parts = [types.Part(text=issue_text)]
                    if include_image and image_bytes is not None:
                        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
                    content = types.Content(role="user", parts=parts)

                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*mismatch.*")
                        runner = InMemoryRunner(agent=adk_agent, app_name="software_bug_assistant")

                    user_id = "web_ui_user"
                    session_id = f"session_{uuid.uuid4().hex[:8]}"

                    await runner.session_service.create_session(
                        app_name="software_bug_assistant",
                        user_id=user_id,
                        session_id=session_id,
                    )

                    response_chunks = []
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=content
                    ):
                        if hasattr(event, 'content') and event.content:
                            if hasattr(event.content, 'parts') and event.content.parts:
                                for response_part in event.content.parts:
                                    if hasattr(response_part, 'text') and response_part.text:
                                        response_chunks.append(response_part.text)
                            elif hasattr(event.content, 'text'):
                                response_chunks.append(event.content.text)
                        elif hasattr(event, 'text'):
                            response_chunks.append(event.text)
                        elif isinstance(event, str):
                            response_chunks.append(event)

                    return ''.join(response_chunks) if response_chunks else None

                try:
                    response_text = asyncio.run(run_agent_async(include_image=True))
                except Exception as image_run_err:
                    if image_bytes is not None:
                        print(f"Cloud triage image processing failed, retrying text-only: {image_run_err}")
                        response_text = asyncio.run(run_agent_async(include_image=False))
                        response_text = (
                            "I couldn't fully process the image this time, so I used your text prompt instead.\n\n"
                            + (response_text or "")
                        )
                    else:
                        raise

                if response_text and response_text.strip():
                    response_text = response_text.strip()
                    try:
                        stack = query_stackexchange(issue_text)
                        stack_section = _format_search_section("📚 StackOverflow Related Issues", stack.get("results", []))
                        if stack_section:
                            response_text += "\n\n---\n\n" + stack_section
                    except Exception as search_err:
                        print(f"StackExchange search skipped: {search_err}")

                    return send_chat_response(
                        response_text,
                        model_used="gemini-cloud",
                        classification="FORCED_CLOUD" if force_cloud else "TRIAGE",
                        ask_fix_confirmation=ask_fix_confirmation,
                        is_stream=False,
                        state=state
                    )

        except Exception as e:
            print(f"Cloud triage error: {e}")
            import traceback
            traceback.print_exc()
            if force_cloud:
                return send_chat_response(
                    f"Cloud analysis failed: {e}. Please retry or ask without `@cloud` to use the local Gemma model.",
                    model_used="gemini-cloud",
                    classification="FORCED_CLOUD",
                    error=True,
                    is_stream=is_stream,
                    state=state
                )
            print("Falling back to local models after cloud failure...")

    # Fallback to local models (when cloud is unavailable or failed)
    if local_llm_with_tools and local_llm_with_tools.is_available():
        try:
            local_tool_response = local_llm_with_tools.chat(issue_text)
            response_text = local_tool_response or ""
            response_text = response_text.strip() or "I can help, but I need more details about the issue."
            return send_chat_response(
                response_text,
                model_used="gemma-2b-tools",
                classification="TRIAGE",
                ask_fix_confirmation=ask_fix_confirmation,
                is_stream=is_stream,
                state=state
            )
        except Exception as local_err:
            print(f"Local LLM with tools fallback failed: {local_err}")

    if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
        try:
            if is_stream:
                gen = hybrid_agent.local_model.generate_stream(issue_text, max_tokens=150, context=context)
                return send_chat_response(
                    gen,
                    model_used="gemma-2b-local",
                    classification="TRIAGE",
                    ask_fix_confirmation=ask_fix_confirmation,
                    is_stream=True,
                    state=state
                )
            else:
                local_response = hybrid_agent.local_model.generate(issue_text, max_tokens=150, context=context)
                if local_response and local_response.strip():
                    return send_chat_response(
                        local_response.strip(),
                        model_used="gemma-2b-local",
                        classification="TRIAGE",
                        ask_fix_confirmation=ask_fix_confirmation,
                        is_stream=False,
                        state=state
                    )
        except Exception as local_err:
            print(f"Local model fallback failed: {local_err}")

    # Final offline fallback if neither cloud nor local model could respond
    response_text = (
        "I'm currently unable to reach the AI models. Please check your connection or provide more details."
    )
    try:
        stack = query_stackexchange(issue_text)
        stack_section = _format_search_section("📚 StackOverflow Related Issues", stack.get("results", []))
        if stack_section:
            response_text += "\n\n---\n\n" + stack_section
    except Exception as search_err:
        print(f"StackExchange search skipped: {search_err}")

    return send_chat_response(
        response_text,
        model_used="offline",
        classification="TRIAGE",
        ask_fix_confirmation=ask_fix_confirmation,
        is_stream=is_stream,
        state=state
    )


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Starting Unified Hybrid AI Web UI")
    print("=" * 70)
    print()
    
    import threading
    import subprocess
    
    def background_download():
        print("Starting background download of Gemma model...")
        try:
            subprocess.run(["uv", "run", "python", "download_gemma.py"], check=False)
            print("Gemma background download finished (or already exists).")
        except Exception as e:
            print(f"Background download error: {e}")
            
    threading.Thread(target=background_download, daemon=True).start()
    
    # Initialize agents at startup
    print("Initializing agents...")
    init_agents()
    
    if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
        print("✓ Local Gemma-2B (Simple): READY")
    else:
        print("⚠ Local Gemma-2B (Simple): NOT AVAILABLE")
    
    if local_llm_with_tools and local_llm_with_tools.is_available():
        print("✓ Local Gemma-2B (Tool Calling): READY")
        print(f"  └── {len(local_llm_with_tools.tool_registry.tools)} tools available")
    else:
        print("⚠ Local Gemma-2B (Tool Calling): NOT AVAILABLE")
    
    if adk_agent:
        print("✓ Cloud Gemini + Tools: READY")
    else:
        print("⚠ Cloud Gemini: NOT AVAILABLE (check GOOGLE_API_KEY and .env)")
    
    print()
    port = int(os.environ.get("PORT", 7860))
    print(f"📍 Access the UI at: http://localhost:{port}")
    print()
    print("Features:")
    print("  • Simple queries → 🏠 Local Gemma-2B (fast, free)")
    print("  • Complex queries → ☁️ Cloud Gemini + Tools")
    print("  • Fallback → 🔧 Local Gemma-2B with Tool Calling")
    print("  • Single unified interface")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    # Suppress Flask/click banner to avoid Windows console error 6
    import click
    click.echo = lambda *a, **kw: None
    
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', port, app, use_reloader=False, use_debugger=False)

