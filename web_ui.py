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
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-void: #06070a;
            --bg-surface: #0c0e14;
            --bg-card: rgba(18, 22, 32, 0.65);
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-card: rgba(255, 255, 255, 0.09);
            
            --accent-mint: #00f59b;
            --accent-mint-glow: rgba(0, 245, 155, 0.35);
            --accent-cyan: #00e5ff;
            --accent-amber: #ffb800;
            --accent-rose: #ff3366;
            
            --text-pure: #ffffff;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-void);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .aurora-bg {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -2;
            background: radial-gradient(circle at 50% 100%, #0a0d16 0%, #06070a 100%);
        }

        .aurora-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.3;
            pointer-events: none;
        }
        .aurora-orb-1 {
            width: 500px; height: 500px;
            background: radial-gradient(circle, #00f59b 0%, rgba(0, 245, 155, 0) 70%);
            top: -100px; left: 20%;
        }
        .aurora-orb-2 {
            width: 550px; height: 550px;
            background: radial-gradient(circle, #00b4d8 0%, rgba(0, 180, 216, 0) 70%);
            bottom: -150px; right: 10%;
        }

        .matrix-overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -1;
            background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px);
            background-size: 32px 32px;
            opacity: 0.5;
        }

        .page {
            max-width: 1050px;
            margin: 0 auto;
            padding: 36px 24px 64px;
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
        }

        .title {
            font-size: 30px;
            font-weight: 800;
            color: var(--text-pure);
            letter-spacing: -0.02em;
        }

        .subtitle {
            color: var(--text-secondary);
            margin-top: 6px;
            font-size: 14px;
        }

        .back-btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-card);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 12px;
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .back-btn:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px);
        }

        .section-header {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--accent-mint);
            margin: 28px 0 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .model-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 18px;
            padding: 24px;
            transition: all 0.25s ease;
        }

        .model-card:hover {
            border-color: rgba(0, 245, 155, 0.4);
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 245, 155, 0.1);
        }

        .model-tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .model-tag.local {
            background: rgba(0, 245, 155, 0.15);
            color: #00f59b;
            border: 1px solid rgba(0, 245, 155, 0.3);
        }

        .model-tag.cloud {
            background: rgba(0, 229, 255, 0.15);
            color: #00e5ff;
            border: 1px solid rgba(0, 229, 255, 0.3);
        }

        .model-card h3 {
            font-size: 19px;
            font-weight: 700;
            color: var(--text-pure);
            margin-bottom: 8px;
        }

        .model-card p {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.55;
            margin-bottom: 16px;
        }

        .syntax-box {
            background: rgba(0, 0, 0, 0.45);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 14px;
            font-size: 12.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .syntax-box code {
            color: #8affcf;
            font-weight: 700;
            background: rgba(0, 245, 155, 0.15);
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
            color: var(--text-primary);
        }

        .cmd-list li code {
            color: #f1f5f9;
            background: rgba(255, 255, 255, 0.06);
            padding: 2px 7px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            border: 1px solid var(--border-subtle);
        }

        .hybrid-banner {
            background: linear-gradient(135deg, rgba(0, 245, 155, 0.1) 0%, rgba(0, 229, 255, 0.05) 100%);
            border: 1px solid rgba(0, 245, 155, 0.25);
            border-radius: 16px;
            padding: 18px 22px;
            margin-bottom: 24px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }

        .hybrid-icon { font-size: 26px; }
        .hybrid-text h4 { font-size: 15px; font-weight: 700; color: #8affcf; margin-bottom: 4px; }
        .hybrid-text p { font-size: 13px; color: var(--text-secondary); line-height: 1.5; margin: 0; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 18px;
        }

        .card h3 {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-pure);
            margin-bottom: 12px;
        }

        .card ul { list-style: none; }
        .card li { font-size: 12.5px; color: var(--text-secondary); padding: 5px 0; }
        .card li::before { content: "• "; color: var(--accent-mint); font-weight: bold; }
    </style>
</head>
<body>
    <div class="aurora-bg">
        <div class="aurora-orb aurora-orb-1"></div>
        <div class="aurora-orb aurora-orb-2"></div>
    </div>
    <div class="matrix-overlay"></div>

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
    <title>Tandion Bug Assistant | Next-Gen AI Issue Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            /* 2026 Cyber Obsidian & Bioluminescent Emerald Palette */
            --bg-void: #06070a;
            --bg-surface: #0c0e14;
            --bg-card: rgba(18, 22, 32, 0.65);
            --bg-card-hover: rgba(26, 32, 48, 0.75);
            
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-card: rgba(255, 255, 255, 0.09);
            --border-highlight: rgba(0, 245, 155, 0.4);
            
            /* High-energy Trending Accents */
            --accent-mint: #00f59b;
            --accent-mint-glow: rgba(0, 245, 155, 0.35);
            --accent-cyan: #00e5ff;
            --accent-cyan-glow: rgba(0, 229, 255, 0.3);
            --accent-amber: #ffb800;
            --accent-rose: #ff3366;
            --accent-violet: #9d4edd;
            
            /* Text Hierarchy */
            --text-pure: #ffffff;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            /* Layout & Transitions */
            --sidebar-w: 275px;
            --header-h: 64px;
            --ease-spring: cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-void);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            letter-spacing: -0.015em;
        }
        
        /* 🌌 Fluid Living Aurora Animations (Floating Mesh Orbs) */
        .aurora-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -2;
            overflow: hidden;
            background: radial-gradient(circle at 50% 100%, #0a0d16 0%, #06070a 100%);
        }
        
        .aurora-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            opacity: 0.35;
            pointer-events: none;
            mix-blend-mode: screen;
            will-change: transform;
        }
        
        .aurora-orb-1 {
            width: 550px;
            height: 550px;
            background: radial-gradient(circle, #00f59b 0%, rgba(0, 245, 155, 0) 70%);
            top: -100px;
            left: 15%;
            animation: orbFloat1 22s infinite ease-in-out alternate;
        }
        
        .aurora-orb-2 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #00b4d8 0%, rgba(0, 180, 216, 0) 70%);
            bottom: -150px;
            right: 10%;
            animation: orbFloat2 28s infinite ease-in-out alternate;
        }
        
        .aurora-orb-3 {
            width: 480px;
            height: 480px;
            background: radial-gradient(circle, #7209b7 0%, rgba(114, 9, 183, 0) 70%);
            top: 40%;
            left: 55%;
            opacity: 0.25;
            animation: orbFloat3 25s infinite ease-in-out alternate;
        }
        
        @keyframes orbFloat1 {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(120px, 80px) scale(1.15); }
            100% { transform: translate(-80px, 140px) scale(0.95); }
        }
        
        @keyframes orbFloat2 {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-140px, -90px) scale(1.2); }
            100% { transform: translate(90px, -130px) scale(0.9); }
        }
        
        @keyframes orbFloat3 {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-100px, 120px) scale(1.1); }
            100% { transform: translate(110px, -70px) scale(0.85); }
        }
        
        /* Modern Micro Dot Grid Matrix */
        .matrix-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px);
            background-size: 32px 32px;
            opacity: 0.6;
            mask-image: radial-gradient(ellipse at 50% 50%, black 40%, transparent 95%);
            -webkit-mask-image: radial-gradient(ellipse at 50% 50%, black 40%, transparent 95%);
        }
        
        /* App Layout Container */
        .app-shell {
            display: flex;
            height: 100vh;
            width: 100%;
            position: relative;
            overflow: hidden;
        }
        
        /* Glass Sidebar */
        .sidebar {
            width: var(--sidebar-w);
            background: rgba(10, 13, 20, 0.75);
            backdrop-filter: blur(28px);
            -webkit-backdrop-filter: blur(28px);
            border-right: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            padding: 20px 16px;
            transition: var(--ease-spring) 0.3s;
            z-index: 40;
            flex-shrink: 0;
        }
        
        .app-shell.sidebar-collapsed .sidebar {
            width: 0;
            padding: 0;
            opacity: 0;
            border-right: none;
            overflow: hidden;
            pointer-events: none;
        }
        
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            margin-bottom: 24px;
            padding: 2px 4px;
        }
        
        .brand-gem {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: linear-gradient(135deg, #00f59b 0%, #00b4d8 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 19px;
            box-shadow: 0 0 20px var(--accent-mint-glow);
            transition: transform 0.25s var(--ease-spring);
            flex-shrink: 0;
        }
        
        .sidebar-brand:hover .brand-gem {
            transform: scale(1.08) rotate(4deg);
        }
        
        .brand-name {
            font-size: 16px;
            font-weight: 800;
            color: var(--text-pure);
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        .brand-badge {
            font-size: 10px;
            font-weight: 700;
            color: var(--accent-mint);
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        
        .sidebar-nav-title {
            font-size: 10.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin: 0 6px 10px;
        }
        
        /* Runtime Engine Glass Cards */
        .engine-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 8px;
            transition: all 0.25s var(--ease-spring);
            position: relative;
            overflow: hidden;
        }
        
        .engine-card:hover {
            background: var(--bg-card-hover);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }
        
        .engine-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .engine-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* ⚡ Radar Pulse Ping Indicator */
        .radar-dot-wrap {
            position: relative;
            width: 8px;
            height: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .radar-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-mint);
            box-shadow: 0 0 10px var(--accent-mint-glow);
            z-index: 2;
        }
        
        .radar-ping {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: var(--accent-mint);
            opacity: 0.8;
            animation: radarPulse 2s cubic-bezier(0, 0, 0.2, 1) infinite;
        }
        
        @keyframes radarPulse {
            0% { transform: scale(1); opacity: 0.8; }
            70% { transform: scale(3); opacity: 0; }
            100% { transform: scale(3.5); opacity: 0; }
        }
        
        .radar-dot.loading {
            background: var(--accent-amber);
            box-shadow: 0 0 10px rgba(255, 184, 0, 0.6);
        }
        
        .radar-dot.offline {
            background: var(--accent-rose);
            box-shadow: 0 0 10px rgba(255, 51, 102, 0.6);
        }
        
        .engine-chip {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 20px;
            letter-spacing: 0.3px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .engine-chip.mint {
            background: rgba(0, 245, 155, 0.14);
            color: #00f59b;
            border: 1px solid rgba(0, 245, 155, 0.25);
        }
        
        .engine-chip.cyan {
            background: rgba(0, 229, 255, 0.14);
            color: #00e5ff;
            border: 1px solid rgba(0, 229, 255, 0.25);
        }
        
        .engine-desc {
            font-size: 11px;
            color: var(--text-secondary);
            line-height: 1.35;
        }
        
        /* Action Buttons */
        .sidebar-actions {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 14px;
        }
        
        .nav-link-btn {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 12px;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s var(--ease-spring);
            text-decoration: none;
            width: 100%;
        }
        
        .nav-link-btn:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-pure);
            border-color: var(--border-subtle);
            transform: translateX(3px);
        }
        
        .nav-link-btn.highlight-link {
            background: linear-gradient(135deg, rgba(0, 245, 155, 0.1) 0%, rgba(0, 229, 255, 0.05) 100%);
            color: #8affcf;
            border-color: rgba(0, 245, 155, 0.2);
            font-weight: 600;
        }
        
        .nav-link-btn.highlight-link:hover {
            background: linear-gradient(135deg, rgba(0, 245, 155, 0.18) 0%, rgba(0, 229, 255, 0.1) 100%);
            color: #ffffff;
            border-color: rgba(0, 245, 155, 0.4);
            box-shadow: 0 0 16px rgba(0, 245, 155, 0.15);
        }
        
        .link-icon {
            font-size: 15px;
            width: 20px;
            text-align: center;
        }
        
        /* Footer Metrics */
        .sidebar-foot {
            margin-top: auto;
            padding-top: 14px;
            border-top: 1px solid var(--border-subtle);
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        
        .metric-cell {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 8px 10px;
            text-align: center;
        }
        
        .metric-val {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-pure);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .metric-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-top: 2px;
        }
        
        /* Main Chat Workspace */
        .workspace-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            min-width: 0;
            position: relative;
        }
        
        /* Header Navigation */
        .workspace-header {
            height: var(--header-h);
            padding: 0 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(6, 7, 10, 0.7);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            z-index: 30;
            flex-shrink: 0;
        }
        
        .header-left-tools {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .icon-toggle-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.2s var(--ease-spring);
        }
        
        .icon-toggle-btn:hover {
            color: var(--text-pure);
            border-color: rgba(255, 255, 255, 0.2);
            background: var(--bg-card-hover);
        }
        
        /* 🎛️ Cyber Segmented Mode Switcher */
        .cyber-segmented-group {
            display: flex;
            align-items: center;
            background: rgba(14, 18, 27, 0.85);
            border: 1px solid var(--border-card);
            padding: 3px;
            border-radius: 14px;
            gap: 2px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }
        
        .cyber-pill {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 12px;
            font-weight: 500;
            padding: 6px 14px;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.25s var(--ease-spring);
            white-space: nowrap;
        }
        
        .cyber-pill:hover {
            color: var(--text-pure);
            background: rgba(255, 255, 255, 0.04);
        }
        
        .cyber-pill.active {
            background: linear-gradient(135deg, rgba(0, 245, 155, 0.95) 0%, rgba(0, 200, 140, 0.95) 100%);
            color: #04100b;
            font-weight: 700;
            box-shadow: 0 2px 14px var(--accent-mint-glow);
        }
        
        .cyber-pill.active.mode-local {
            background: linear-gradient(135deg, #00f59b 0%, #00b4d8 100%);
            color: #04100b;
            box-shadow: 0 2px 14px var(--accent-mint-glow);
        }
        
        .cyber-pill.active.mode-cloud {
            background: linear-gradient(135deg, #00e5ff 0%, #3a86ff 100%);
            color: #020b14;
            box-shadow: 0 2px 14px var(--accent-cyan-glow);
        }
        
        .header-right-tools {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .tool-pill-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            padding: 6px 13px;
            border-radius: 10px;
            font-size: 12.5px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s var(--ease-spring);
        }
        
        .tool-pill-btn:hover {
            color: var(--text-pure);
            background: var(--bg-card-hover);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        
        /* Chat Feed */
        .chat-scrollway {
            flex: 1;
            overflow-y: auto;
            padding: 28px 24px 130px;
            display: flex;
            flex-direction: column;
            gap: 22px;
            scroll-behavior: smooth;
        }
        
        .chat-scrollway::-webkit-scrollbar {
            width: 5px;
        }
        
        .chat-scrollway::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-scrollway::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        /* ⚡ Hero Welcome Section */
        .welcome-center {
            max-width: 820px;
            margin: auto;
            padding: 30px 20px 10px;
            text-align: center;
            animation: heroFadeIn 0.5s var(--ease-spring) forwards;
        }
        
        @keyframes heroFadeIn {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Levitating Animated Hero Badge */
        .floating-hero-avatar {
            width: 72px;
            height: 72px;
            margin: 0 auto 22px;
            background: linear-gradient(135deg, #00f59b 0%, #00b4d8 50%, #7209b7 100%);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 34px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 0 35px var(--accent-mint-glow);
            animation: levitateBadge 4s ease-in-out infinite alternate;
        }
        
        @keyframes levitateBadge {
            0% { transform: translateY(0) rotate(0deg); box-shadow: 0 0 25px var(--accent-mint-glow); }
            50% { transform: translateY(-7px) rotate(2deg); box-shadow: 0 10px 40px rgba(0, 229, 255, 0.45); }
            100% { transform: translateY(-3px) rotate(-1deg); box-shadow: 0 5px 30px var(--accent-mint-glow); }
        }
        
        .hero-head-title {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .hero-head-desc {
            font-size: 14.5px;
            color: var(--text-secondary);
            max-width: 580px;
            margin: 0 auto 32px;
            line-height: 1.6;
        }
        
        /* 🎴 Shimmering Interactive Prompt Cards */
        .prompt-cards-matrix {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 14px;
            text-align: left;
        }
        
        .interactive-prompt-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 18px;
            cursor: pointer;
            transition: all 0.3s var(--ease-spring);
            position: relative;
            overflow: hidden;
        }
        
        /* Card Light Shimmer Sweep */
        .interactive-prompt-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 50%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent);
            transform: skewX(-20deg);
            transition: left 0.6s ease;
        }
        
        .interactive-prompt-card:hover::after {
            left: 200%;
        }
        
        .interactive-prompt-card:hover {
            border-color: var(--border-highlight);
            transform: translateY(-4px);
            background: var(--bg-card-hover);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 245, 155, 0.15);
        }
        
        .card-top-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        
        .card-gem-icon {
            font-size: 18px;
        }
        
        .card-caption {
            font-size: 14.5px;
            font-weight: 700;
            color: var(--text-pure);
        }
        
        .card-explainer {
            font-size: 12.5px;
            color: var(--text-secondary);
            line-height: 1.45;
        }
        
        .card-query-chip {
            display: inline-block;
            font-size: 10.5px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            margin-top: 12px;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.45);
            color: #8affcf;
            border: 1px solid rgba(0, 245, 155, 0.2);
        }
        
        /* Message Stream Rows */
        .stream-message-row {
            display: flex;
            gap: 14px;
            max-width: 860px;
            width: 100%;
            margin: 0 auto;
            animation: messageSpringIn 0.3s var(--ease-spring) forwards;
        }
        
        @keyframes messageSpringIn {
            from { opacity: 0; transform: translateY(10px) scale(0.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        .stream-message-row.user {
            justify-content: flex-end;
        }
        
        .msg-avatar {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .stream-message-row.assistant .msg-avatar {
            background: linear-gradient(135deg, #00f59b 0%, #00b4d8 100%);
            color: #04100b;
            font-weight: 800;
            box-shadow: 0 0 16px var(--accent-mint-glow);
        }
        
        .stream-message-row.user .msg-avatar {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-card);
            order: 2;
        }
        
        .msg-content-wrapper {
            max-width: calc(100% - 50px);
            display: flex;
            flex-direction: column;
        }
        
        .stream-message-row.user .msg-content-wrapper {
            align-items: flex-end;
            order: 1;
        }
        
        /* Message Bubbles */
        .speech-bubble {
            border-radius: 18px;
            padding: 14px 18px;
            font-size: 14px;
            line-height: 1.6;
            word-break: break-word;
            position: relative;
        }
        
        .stream-message-row.user .speech-bubble {
            background: linear-gradient(135deg, #00f59b 0%, #00b4d8 100%);
            color: #04100b;
            font-weight: 500;
            border-bottom-right-radius: 4px;
            box-shadow: 0 6px 20px rgba(0, 245, 155, 0.25);
        }
        
        .stream-message-row.assistant .speech-bubble {
            background: rgba(14, 18, 27, 0.85);
            border: 1px solid var(--border-card);
            border-bottom-left-radius: 4px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            color: var(--text-primary);
        }
        
        /* Message Action Toolbar */
        .msg-action-toolbar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 6px;
            padding: 0 4px;
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .origin-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 10.5px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .origin-badge.local {
            background: rgba(0, 245, 155, 0.12);
            color: #00f59b;
            border: 1px solid rgba(0, 245, 155, 0.25);
        }
        
        .origin-badge.cloud {
            background: rgba(0, 229, 255, 0.12);
            color: #00e5ff;
            border: 1px solid rgba(0, 229, 255, 0.25);
        }
        
        .copy-action-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 11px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 6px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        
        .copy-action-btn:hover {
            color: var(--text-pure);
            background: rgba(255, 255, 255, 0.08);
        }
        
        /* Code Block Styling */
        .speech-bubble p { margin-bottom: 12px; }
        .speech-bubble p:last-child { margin-bottom: 0; }
        .speech-bubble ul, .speech-bubble ol { margin: 8px 0 12px 20px; }
        .speech-bubble li { margin-bottom: 4px; }
        
        .speech-bubble blockquote {
            border-left: 3px solid var(--accent-mint);
            padding: 6px 14px;
            margin: 10px 0;
            background: rgba(0, 245, 155, 0.06);
            border-radius: 0 8px 8px 0;
            color: #d1fae5;
        }
        
        .speech-bubble code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12.5px;
            background: rgba(0, 0, 0, 0.45);
            padding: 2px 6px;
            border-radius: 6px;
            color: #6ee7b7;
            border: 1px solid rgba(0, 245, 155, 0.15);
        }
        
        .code-container-box {
            margin: 12px 0;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--border-card);
            background: rgba(4, 6, 10, 0.85);
        }
        
        .code-title-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 7px 14px;
            background: rgba(16, 21, 32, 0.95);
            border-bottom: 1px solid var(--border-subtle);
            font-size: 11px;
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
        }
        
        .copy-terminal-btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 11px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.2s ease;
        }
        
        .copy-terminal-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            color: var(--text-pure);
        }
        
        .speech-bubble pre {
            margin: 0;
            padding: 14px;
            overflow-x: auto;
            background: transparent;
        }
        
        .speech-bubble pre code {
            background: transparent;
            padding: 0;
            border: none;
            color: #f1f5f9;
            font-size: 12.5px;
            line-height: 1.5;
        }
        
        .table-responsive-box {
            overflow-x: auto;
            margin: 12px 0;
            border-radius: 12px;
            border: 1px solid var(--border-card);
        }
        
        .speech-bubble table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
        }
        
        .speech-bubble th {
            background: rgba(18, 24, 36, 0.95);
            color: var(--text-pure);
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-card);
            font-weight: 700;
        }
        
        .speech-bubble td {
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-secondary);
        }
        
        .speech-bubble tr:last-child td { border-bottom: none; }
        .speech-bubble tr:nth-child(even) { background: rgba(255, 255, 255, 0.02); }
        
        /* Typing Wave Dots */
        .typing-wave-bubble {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 12px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            border-bottom-left-radius: 4px;
        }
        
        .wave-dot {
            width: 7px;
            height: 7px;
            background: var(--accent-mint);
            border-radius: 50%;
            animation: waveBounce 1.4s infinite ease-in-out both;
        }
        
        .wave-dot:nth-child(1) { animation-delay: -0.32s; }
        .wave-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes waveBounce {
            0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
            40% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 8px var(--accent-mint-glow); }
        }
        
        .streaming-cursor {
            display: inline-block;
            width: 7px;
            height: 15px;
            background: var(--accent-mint);
            margin-left: 4px;
            vertical-align: text-bottom;
            animation: blinkCursor 0.9s infinite;
        }
        
        @keyframes blinkCursor {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        
        /* 🕹️ Floating Cyber Command Console */
        .console-dock {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0 24px 20px;
            background: linear-gradient(to top, var(--bg-void) 65%, rgba(6, 7, 10, 0.8) 85%, transparent 100%);
            pointer-events: none;
            z-index: 25;
        }
        
        .console-dock-inner {
            max-width: 860px;
            margin: 0 auto;
            pointer-events: auto;
        }
        
        .dock-pill-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 0 4px;
        }
        
        .dock-mode-chips {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .mode-chip-tag {
            background: rgba(16, 21, 32, 0.7);
            border: 1px solid var(--border-card);
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 500;
            padding: 3px 10px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s var(--ease-spring);
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .mode-chip-tag:hover {
            color: var(--text-pure);
            border-color: rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.06);
        }
        
        .mode-chip-tag.active {
            color: #04100b;
            background: var(--accent-mint);
            border-color: var(--accent-mint);
            font-weight: 700;
            box-shadow: 0 0 12px var(--accent-mint-glow);
        }
        
        /* Input Box */
        .cyber-input-wrapper {
            background: rgba(14, 18, 28, 0.85);
            backdrop-filter: blur(28px);
            -webkit-backdrop-filter: blur(28px);
            border: 1px solid var(--border-card);
            border-radius: 18px;
            padding: 6px 8px 6px 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            transition: all 0.3s var(--ease-spring);
        }
        
        .cyber-input-wrapper:focus-within {
            border-color: rgba(0, 245, 155, 0.65);
            box-shadow: 0 8px 32px rgba(0, 245, 155, 0.15), 0 0 0 2px rgba(0, 245, 155, 0.2);
        }
        
        .dock-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-pure);
            font-size: 14px;
            font-family: inherit;
            padding: 8px 0;
        }
        
        .dock-input::placeholder {
            color: var(--text-muted);
        }
        
        .dock-action-btn {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: all 0.25s var(--ease-spring);
            flex-shrink: 0;
        }
        
        .btn-camera {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            border: 1px solid var(--border-card);
        }
        
        .btn-camera:hover {
            color: var(--text-pure);
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.25);
            transform: scale(1.05);
        }
        
        .btn-launch {
            background: linear-gradient(135deg, #00f59b 0%, #00b4d8 100%);
            color: #04100b;
            font-weight: 700;
            width: auto;
            padding: 0 18px;
            gap: 6px;
            font-size: 13.5px;
            box-shadow: 0 2px 14px var(--accent-mint-glow);
        }
        
        .btn-launch:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 22px var(--accent-mint-glow);
        }
        
        .btn-launch:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-halt {
            display: none;
            background: rgba(255, 51, 102, 0.15);
            border: 1px solid rgba(255, 51, 102, 0.35);
            color: #ff99b3;
            font-weight: 700;
            width: auto;
            padding: 0 14px;
            gap: 6px;
            font-size: 13px;
        }
        
        .btn-halt.visible {
            display: flex;
        }
        
        .btn-halt:hover {
            background: rgba(255, 51, 102, 0.3);
            color: #ffffff;
        }
        
        .halt-box {
            width: 10px;
            height: 10px;
            background: currentColor;
            border-radius: 2px;
        }
        
        /* Attachment Preview */
        .dock-tray {
            display: none;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            padding: 6px 12px;
            background: rgba(14, 18, 28, 0.85);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            max-width: fit-content;
        }
        
        .dock-tray.visible { display: flex; }
        
        .tray-preview-img {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            object-fit: cover;
            border: 1px solid var(--border-card);
        }
        
        .tray-caption {
            font-size: 12px;
            color: var(--text-secondary);
            max-width: 220px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .tray-remove-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 14px;
            line-height: 1;
            padding: 2px;
        }
        
        .tray-remove-btn:hover { color: var(--accent-rose); }
        
        .dock-shortcut-caption {
            text-align: center;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 6px;
        }
        
        .dock-shortcut-caption kbd {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            padding: 1px 4px;
            color: var(--text-secondary);
        }
        
        /* Mobile Overlay */
        .drawer-dimmer {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(6px);
            z-index: 35;
        }
        
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                height: 100%;
                transform: translateX(-100%);
                box-shadow: 0 0 35px rgba(0, 0, 0, 0.8);
            }
            
            .app-shell.mobile-drawer-open .sidebar {
                transform: translateX(0);
            }
            
            .app-shell.mobile-drawer-open .drawer-dimmer {
                display: block;
            }
            
            .cyber-segmented-group { display: none; }
            .chat-scrollway { padding: 16px 14px 130px; }
            .console-dock { padding: 0 12px 14px; }
            .prompt-cards-matrix { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <!-- 🌌 Living Aurora Background Orbs -->
    <div class="aurora-bg">
        <div class="aurora-orb aurora-orb-1"></div>
        <div class="aurora-orb aurora-orb-2"></div>
        <div class="aurora-orb aurora-orb-3"></div>
    </div>
    <div class="matrix-overlay"></div>
    <div class="drawer-dimmer" id="drawerDimmer" onclick="toggleMobileSidebar(false)"></div>
    
    <div class="app-shell" id="appShell">
        <!-- Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <a href="/" class="sidebar-brand">
                <div class="brand-gem">⚡</div>
                <div>
                    <div class="brand-name">Tandion</div>
                    <div class="brand-badge">Bug Assistant</div>
                </div>
            </a>
            
            <!-- Live Model Engines -->
            <div class="sidebar-nav-title">AI Engines</div>
            <div class="engine-card">
                <div class="engine-header">
                    <span class="engine-label">
                        <span class="radar-dot-wrap">
                            <span class="radar-dot" id="localDot"></span>
                            <span class="radar-ping"></span>
                        </span>
                        Local Engine
                    </span>
                    <span class="engine-chip mint" id="localBadge">Gemma-2B</span>
                </div>
                <div class="engine-desc">100% Offline • Zero-latency CPU • Free</div>
            </div>
            
            <div class="engine-card">
                <div class="engine-header">
                    <span class="engine-label">
                        <span class="radar-dot-wrap">
                            <span class="radar-dot" id="cloudDot"></span>
                            <span class="radar-ping"></span>
                        </span>
                        Cloud Engine
                    </span>
                    <span class="engine-chip cyan" id="cloudBadge">Gemini</span>
                </div>
                <div class="engine-desc">Deep Reasoning • Live Tools • Search</div>
            </div>
            
            <!-- Quick Actions -->
            <div class="sidebar-nav-title" style="margin-top: 16px;">Quick Actions</div>
            <div class="sidebar-actions">
                <a href="/help" class="nav-link-btn highlight-link">
                    <span class="link-icon">💡</span>
                    <span>Help & Examples</span>
                </a>
                <button class="nav-link-btn" onclick="quickAction('Show all open tickets')">
                    <span class="link-icon">📋</span>
                    <span>Open Tickets</span>
                </button>
                <button class="nav-link-btn" onclick="quickAction('Show critical priority tickets')">
                    <span class="link-icon">🚨</span>
                    <span>Critical Issues</span>
                </button>
                <button class="nav-link-btn" onclick="quickAction('Show high priority tickets')">
                    <span class="link-icon">⚠️</span>
                    <span>High Priority</span>
                </button>
                <button class="nav-link-btn" onclick="quickAction('Show closed tickets')">
                    <span class="link-icon">✅</span>
                    <span>Closed Tickets</span>
                </button>
            </div>
            
            <!-- Metrics Footer -->
            <div class="sidebar-foot">
                <div class="metric-grid">
                    <div class="metric-cell">
                        <div class="metric-val" id="queryCount">0</div>
                        <div class="metric-label">Queries</div>
                    </div>
                    <div class="metric-cell">
                        <div class="metric-val" id="sessionTime">0m</div>
                        <div class="metric-label">Session</div>
                    </div>
                </div>
            </div>
        </aside>
        
        <!-- Main Workspace -->
        <main class="workspace-panel">
            <!-- Header Bar -->
            <header class="workspace-header">
                <div class="header-left-tools">
                    <button class="icon-toggle-btn" id="sidebarToggle" onclick="handleSidebarToggle()" title="Toggle Sidebar">
                        ☰
                    </button>
                    
                    <!-- Segmented Mode Control -->
                    <div class="cyber-segmented-group" id="modeControl">
                        <button class="cyber-pill active" id="btnModeAuto" onclick="setModelMode('auto')" title="Smart Hybrid Routing">
                            ⚡ Auto Hybrid
                        </button>
                        <button class="cyber-pill" id="btnModeLocal" onclick="setModelMode('local')" title="Force Local Offline Gemma-2B">
                            🏠 Local Gemma
                        </button>
                        <button class="cyber-pill" id="btnModeCloud" onclick="setModelMode('cloud')" title="Force Cloud Gemini">
                            ☁️ Cloud Gemini
                        </button>
                    </div>
                </div>
                
                <div class="header-right-tools">
                    <a href="/help" class="tool-pill-btn" title="View documentation">
                        <span>💡</span>
                        <span>Guide</span>
                    </a>
                    <button class="tool-pill-btn" onclick="clearChat()" title="Clear conversation history">
                        <span>🗑️</span>
                        <span>Clear</span>
                    </button>
                    <button class="tool-pill-btn" onclick="refreshStatus()" title="Refresh connection status">
                        <span>🔄</span>
                    </button>
                </div>
            </header>
            
            <!-- Chat Feed Area -->
            <div class="chat-scrollway" id="chatArea">
                <!-- Welcome Card -->
                <div class="welcome-center" id="welcomeCard">
                    <div class="floating-hero-avatar">⚡</div>
                    <h2 class="hero-head-title">Tandion Bug Assistant</h2>
                    <p class="hero-head-desc">
                        High-performance AI bug triage, ticket operations, and stack trace debugging.
                        Choose your routing engine or select an interactive prompt below to begin.
                    </p>
                    
                    <div class="prompt-cards-matrix">
                        <div class="interactive-prompt-card" onclick="quickAction('Show critical priority tickets')">
                            <div class="card-top-row">
                                <span class="card-gem-icon">🚨</span>
                                <span class="card-caption">Critical Bug Triage</span>
                            </div>
                            <div class="card-explainer">Find blocking P0/Critical issues requiring immediate fix.</div>
                            <span class="card-query-chip">Show critical priority tickets</span>
                        </div>
                        
                        <div class="interactive-prompt-card" onclick="quickAction('@local how to debug memory leaks in Python')">
                            <div class="card-top-row">
                                <span class="card-gem-icon">🏠</span>
                                <span class="card-caption">Local Fast Debug</span>
                            </div>
                            <div class="card-explainer">Private offline CPU analysis for Python memory leak diagnosis.</div>
                            <span class="card-query-chip">@local debug memory leaks</span>
                        </div>
                        
                        <div class="interactive-prompt-card" onclick="quickAction('@cloud investigate CVE-2024-3094 vulnerability')">
                            <div class="card-top-row">
                                <span class="card-gem-icon">☁️</span>
                                <span class="card-caption">Cloud CVE & Security</span>
                            </div>
                            <div class="card-explainer">Online deep research with live web tools and vulnerability insights.</div>
                            <span class="card-query-chip">@cloud investigate CVE-2024-3094</span>
                        </div>
                        
                        <div class="interactive-prompt-card" onclick="quickAction('Show all open tickets')">
                            <div class="card-top-row">
                                <span class="card-gem-icon">📋</span>
                                <span class="card-caption">Ticket Dashboard</span>
                            </div>
                            <div class="card-explainer">Query active tickets, assignee distribution, and priority overview.</div>
                            <span class="card-query-chip">Show all open tickets</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Console Dock (Bottom Input) -->
            <div class="console-dock">
                <div class="console-dock-inner">
                    <div class="dock-pill-bar">
                        <div class="dock-mode-chips">
                            <span style="font-size: 11px; color: var(--text-muted); margin-right: 2px;">Mode:</span>
                            <button class="mode-chip-tag active" id="chipAuto" onclick="setModelMode('auto')">⚡ Auto</button>
                            <button class="mode-chip-tag" id="chipLocal" onclick="setModelMode('local')">🏠 @local</button>
                            <button class="mode-chip-tag" id="chipCloud" onclick="setModelMode('cloud')">☁️ @cloud</button>
                        </div>
                        
                        <div class="dock-tray" id="dockTray">
                            <img id="dockThumb" class="tray-preview-img" alt="Attached Preview" />
                            <span id="dockAttachName" class="tray-caption">Screenshot attached</span>
                            <button type="button" class="tray-remove-btn" onclick="clearPendingImage()" title="Remove attachment">✕</button>
                        </div>
                    </div>
                    
                    <div class="cyber-input-wrapper">
                        <input 
                            type="text" 
                            class="dock-input" 
                            id="messageInput"
                            placeholder="Ask about tickets, bugs, or code (Auto Hybrid)..."
                            onkeydown="handleKeyPress(event)"
                            autocomplete="off"
                        />
                        <input type="file" id="imageInput" accept="image/*" style="display:none" />
                        
                        <button class="dock-action-btn btn-camera" id="attachButton" onclick="document.getElementById('imageInput').click()" title="Attach Error Screenshot">
                            📷
                        </button>
                        
                        <button class="dock-action-btn btn-launch" id="sendButton" onclick="sendMessage()">
                            <span>Send</span>
                            <span>→</span>
                        </button>
                        
                        <button class="dock-action-btn btn-halt" id="stopButton" onclick="stopResponse()">
                            <div class="halt-box"></div>
                            <span>Stop</span>
                        </button>
                    </div>
                    
                    <div class="dock-shortcut-caption">
                        Press <kbd>Enter ↵</kbd> to send • <kbd>Shift+Enter</kbd> for newline • Paste screenshots directly from clipboard
                    </div>
                </div>
            </div>
        </main>
    </div>
    
    <script>
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
        let currentMode = 'auto';
        
        let pendingImageBase64 = null;
        let pendingImageMime = null;
        let pendingImageName = null;
        let pendingImagePreviewUrl = null;

        function setModelMode(mode) {
            currentMode = mode;
            const input = document.getElementById('messageInput');
            
            const btnAuto = document.getElementById('btnModeAuto');
            const btnLocal = document.getElementById('btnModeLocal');
            const btnCloud = document.getElementById('btnModeCloud');
            if (btnAuto) btnAuto.className = 'cyber-pill' + (mode === 'auto' ? ' active' : '');
            if (btnLocal) btnLocal.className = 'cyber-pill' + (mode === 'local' ? ' active mode-local' : '');
            if (btnCloud) btnCloud.className = 'cyber-pill' + (mode === 'cloud' ? ' active mode-cloud' : '');
            
            const chipAuto = document.getElementById('chipAuto');
            const chipLocal = document.getElementById('chipLocal');
            const chipCloud = document.getElementById('chipCloud');
            if (chipAuto) chipAuto.className = 'mode-chip-tag' + (mode === 'auto' ? ' active' : '');
            if (chipLocal) chipLocal.className = 'mode-chip-tag' + (mode === 'local' ? ' active' : '');
            if (chipCloud) chipCloud.className = 'mode-chip-tag' + (mode === 'cloud' ? ' active' : '');
            
            if (input) {
                if (mode === 'local') {
                    input.placeholder = 'Ask offline model (forces @local)...';
                } else if (mode === 'cloud') {
                    input.placeholder = 'Ask cloud model with tools (forces @cloud)...';
                } else {
                    input.placeholder = 'Ask about tickets, bugs, or code (Auto Hybrid)...';
                }
            }
        }

        function handleSidebarToggle() {
            const container = document.getElementById('appShell');
            if (window.innerWidth <= 768) {
                toggleMobileSidebar(!container.classList.contains('mobile-drawer-open'));
            } else {
                container.classList.toggle('sidebar-collapsed');
                safeSetItem('local', 'tandion_sidebar_collapsed', container.classList.contains('sidebar-collapsed') ? '1' : '0');
            }
        }

        function toggleMobileSidebar(open) {
            const container = document.getElementById('appShell');
            if (open) {
                container.classList.add('mobile-drawer-open');
            } else {
                container.classList.remove('mobile-drawer-open');
            }
        }

        if (window.innerWidth > 768 && safeGetItem('local', 'tandion_sidebar_collapsed') === '1') {
            document.getElementById('appShell').classList.add('sidebar-collapsed');
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

        function updateAttachIndicator() {
            const tray = document.getElementById('dockTray');
            const thumb = document.getElementById('dockThumb');
            const nameEl = document.getElementById('dockAttachName');
            
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
                <div class="welcome-center" id="welcomeCard">
                    <div class="floating-hero-avatar">⚡</div>
                    <h2 class="hero-head-title">Tandion Bug Assistant</h2>
                    <p class="hero-head-desc">
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
                        localDot.className = 'radar-dot';
                        if (localBadge) localBadge.textContent = 'Gemma-2B';
                    } else if (data.local_downloading) {
                        localDot.className = 'radar-dot loading';
                        if (localBadge) localBadge.textContent = 'Loading...';
                    } else {
                        localDot.className = 'radar-dot offline';
                        if (localBadge) localBadge.textContent = 'Offline';
                    }
                    
                    const cloudDot = document.getElementById('cloudDot');
                    if (cloudDot) {
                        cloudDot.className = 'radar-dot' + (data.cloud_available ? '' : ' offline');
                    }
                })
                .catch(err => console.log('Status refresh error:', err));
        }

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

        function formatCodeBlocks(container) {
            const pres = container.querySelectorAll('pre');
            pres.forEach(pre => {
                if (pre.parentNode && pre.parentNode.classList.contains('code-container-box')) return;
                
                const wrapper = document.createElement('div');
                wrapper.className = 'code-container-box';
                
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
                header.className = 'code-title-bar';
                header.innerHTML = `
                    <span>${lang}</span>
                    <button class="copy-terminal-btn" type="button">📋 Copy</button>
                `;
                
                const copyBtn = header.querySelector('.copy-terminal-btn');
                copyBtn.addEventListener('click', () => {
                    const text = code ? code.innerText : pre.innerText;
                    copyText(text, copyBtn);
                });
                
                pre.parentNode.insertBefore(wrapper, pre);
                wrapper.appendChild(header);
                wrapper.appendChild(pre);
            });
            
            const tables = container.querySelectorAll('table');
            tables.forEach(table => {
                if (table.parentNode && table.parentNode.classList.contains('table-responsive-box')) return;
                const tblWrap = document.createElement('div');
                tblWrap.className = 'table-responsive-box';
                table.parentNode.insertBefore(tblWrap, table);
                tblWrap.appendChild(table);
            });
        }
        
        function addMessage(text, isUser, modelUsed = null, imageUrl = null) {
            const chatArea = document.getElementById('chatArea');
            const welcome = document.getElementById('welcomeCard');
            if (welcome) welcome.remove();
            
            const messageRow = document.createElement('div');
            messageRow.className = `stream-message-row ${isUser ? 'user' : 'assistant'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'msg-avatar';
            avatar.textContent = isUser ? '👤' : '⚡';
            
            const contentWrap = document.createElement('div');
            contentWrap.className = 'msg-content-wrapper';
            
            const bubble = document.createElement('div');
            bubble.className = 'speech-bubble';
            
            if (isUser) {
                if (imageUrl) {
                    const img = document.createElement('img');
                    img.src = imageUrl;
                    img.alt = 'Uploaded screenshot';
                    img.style.maxWidth = '300px';
                    img.style.maxHeight = '240px';
                    img.style.borderRadius = '12px';
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
            
            contentWrap.appendChild(bubble);
            
            if (!isUser) {
                const metaBar = document.createElement('div');
                metaBar.className = 'msg-action-toolbar';
                
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
                    <span class="origin-badge ${tagClass}">
                        ${modelIcon} ${modelLabel}
                    </span>
                    <span>${new Date().toLocaleTimeString()}</span>
                    <button class="copy-action-btn" type="button" title="Copy response">
                        📋 Copy
                    </button>
                `;
                
                const copyBtn = metaBar.querySelector('.copy-action-btn');
                copyBtn.addEventListener('click', () => {
                    copyText(text, copyBtn);
                });
                
                contentWrap.appendChild(metaBar);
            }
            
            messageRow.appendChild(avatar);
            messageRow.appendChild(contentWrap);
            chatArea.appendChild(messageRow);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendButton');
            const chatArea = document.getElementById('chatArea');
            
            let message = input.value.trim();
            if (!message && !pendingImageBase64) return;

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
            
            const stopBtn = document.getElementById('stopButton');
            stopBtn.classList.add('visible');
            isGenerating = true;
            
            currentAbortController = new AbortController();
            
            const typingRow = document.createElement('div');
            typingRow.className = 'stream-message-row assistant';
            typingRow.id = 'typingRow';
            typingRow.innerHTML = `
                <div class="msg-avatar">⚡</div>
                <div class="msg-content-wrapper">
                    <div class="typing-wave-bubble">
                        <div class="wave-dot"></div>
                        <div class="wave-dot"></div>
                        <div class="wave-dot"></div>
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
                                        assistantRow.className = 'stream-message-row assistant';

                                        const avatar = document.createElement('div');
                                        avatar.className = 'msg-avatar';
                                        avatar.textContent = '⚡';

                                        assistantWrapper = document.createElement('div');
                                        assistantWrapper.className = 'msg-content-wrapper';

                                        assistantBubble = document.createElement('div');
                                        assistantBubble.className = 'speech-bubble';
                                        assistantBubble.innerHTML = '<span class="streaming-cursor"></span>';

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
                                        assistantRow.className = 'stream-message-row assistant';

                                        const avatar = document.createElement('div');
                                        avatar.className = 'msg-avatar';
                                        avatar.textContent = '⚡';

                                        assistantWrapper = document.createElement('div');
                                        assistantWrapper.className = 'msg-content-wrapper';

                                        assistantBubble = document.createElement('div');
                                        assistantBubble.className = 'speech-bubble';
                                        assistantWrapper.appendChild(assistantBubble);
                                        assistantRow.appendChild(avatar);
                                        assistantRow.appendChild(assistantWrapper);
                                        chatArea.appendChild(assistantRow);
                                    }

                                    accumulatedText += event.text;
                                    assistantBubble.innerHTML = renderMarkdown(accumulatedText) + '<span class="streaming-cursor"></span>';
                                    formatCodeBlocks(assistantBubble);
                                    chatArea.scrollTop = chatArea.scrollHeight;
                                } else if (event.type === 'done') {
                                    if (assistantBubble) {
                                        assistantBubble.innerHTML = renderMarkdown(accumulatedText);
                                        formatCodeBlocks(assistantBubble);

                                        if (assistantWrapper && !assistantWrapper.querySelector('.msg-action-toolbar')) {
                                            const metaBar = document.createElement('div');
                                            metaBar.className = 'msg-action-toolbar';
                                            
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
                                                <span class="origin-badge ${tagClass}">
                                                    ${modelIcon} ${modelLabel}
                                                </span>
                                                <span>${new Date().toLocaleTimeString()}</span>
                                                <button class="copy-action-btn" type="button" title="Copy response">
                                                    📋 Copy
                                                </button>
                                            `;
                                            
                                            const copyBtn = metaBar.querySelector('.copy-action-btn');
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

