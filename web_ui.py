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

from flask import Flask, render_template_string, request, jsonify, stream_with_context, Response

from software_bug_assistant.tools.local_tools import (
    backfill_missing_embeddings,
    get_ticket_creator,
    query_stackexchange,
    search_google,
)
from software_bug_assistant.hybrid_agent import HybridAgent, classify_query

# Import local LLM with tool calling (Option B - full tool calling)
local_llm_with_tools = None
try:
    from software_bug_assistant.tools.local_tools import get_local_llm_with_tools
    local_llm_with_tools = get_local_llm_with_tools()
    if local_llm_with_tools and local_llm_with_tools.is_available():
        print("✓ Local LLM with Tool Calling: READY")
    else:
        print("⚠ Local LLM with Tool Calling: Model not loaded")
        local_llm_with_tools = None
except Exception as e:
    print(f"⚠ Local LLM with tools not available: {e}")
    local_llm_with_tools = None

# Initialize toolbox client for direct database access (fallback)
# Tool indices based on tools.yaml toolset order:
# 0: search_tickets(query)
# 1: get_ticket_by_id(ticket_id)
# 2: get_tickets_by_assignee(assignee)
# 3: get_tickets_by_status(status)
# 4: get_tickets_by_priority(priority)
# 5: get_tickets_by_date_range(start_date, end_date, date_field)
# 6: update_ticket_priority(priority, ticket_id)
# 7: update_ticket_status(status, ticket_id)
# 8: create_ticket(title, description, assignee, priority, status)
toolbox_tools_list = []
toolbox_tools_by_name = {}
try:
    from toolbox_core import ToolboxSyncClient
    TOOLBOX_URL = os.getenv("MCP_TOOLBOX_URL", "http://127.0.0.1:5000")
    toolbox_client = ToolboxSyncClient(TOOLBOX_URL)
    toolbox_tools_list = toolbox_client.load_toolset("tickets_toolset")
    toolbox_tools_by_name = {getattr(t, '__name__', ''): t for t in toolbox_tools_list}
    print(f"✓ Direct database tools loaded: {len(toolbox_tools_list)} tools")
    print(f"  Tool names: {list(toolbox_tools_by_name.keys())}")
except Exception as e:
    print(f"Toolbox client not available for fallback: {e}")


def query_database_directly(query: str) -> Optional[str]:
    """
    Directly query the database using toolbox when cloud is unavailable.
    Returns formatted results or None if can't handle.
    """
    global toolbox_tools_by_name
    
    if not toolbox_tools_by_name:
        return None
    
    # Get tools by name (order-independent)
    status_tool = toolbox_tools_by_name.get('get-tickets-by-status')
    priority_tool = toolbox_tools_by_name.get('get-tickets-by-priority')
    
    if not status_tool or not priority_tool:
        return None
    
    query_lower = query.lower()
    
    try:
        result = None
        
        # Detect query type and call appropriate tool by name
        if "open" in query_lower and "ticket" in query_lower:
            result = status_tool(status="open")
        elif "closed" in query_lower and "ticket" in query_lower:
            result = status_tool(status="closed")
        elif "in progress" in query_lower or "in-progress" in query_lower:
            result = status_tool(status="in_progress")
        elif "all ticket" in query_lower or "show ticket" in query_lower or "list ticket" in query_lower:
            result = status_tool(status="open")
        elif "high" in query_lower and ("priority" in query_lower or "ticket" in query_lower):
            result = priority_tool(priority="high")
        elif "critical" in query_lower:
            result = priority_tool(priority="critical")
        elif "medium" in query_lower and "priority" in query_lower:
            result = priority_tool(priority="medium")
        elif "low" in query_lower and "priority" in query_lower:
            result = priority_tool(priority="low")
        else:
            return None
        
        if result is None:
            return None
        
        # Parse JSON result and format as markdown table
        import json
        
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                data = json.loads(result)
            except json.JSONDecodeError:
                # If not JSON, check if it looks like a Python list repr
                if result.startswith('[{') or result.startswith('[{'):
                    try:
                        import ast
                        data = ast.literal_eval(result)
                    except:
                        return f"📋 **Database Results** (via local fallback):\n\n{result}"
                else:
                    return f"📋 **Database Results** (via local fallback):\n\n{result}"
        else:
            data = result
        
        # Format as markdown table
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Select key columns for display
            display_cols = ['ticket_id', 'title', 'status', 'priority', 'assignee']
            available_cols = [c for c in display_cols if c in data[0]]
            
            if not available_cols:
                available_cols = list(data[0].keys())[:5]  # First 5 columns
            
            # Build markdown table
            md_table = "| " + " | ".join(available_cols) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(available_cols)) + " |\n"
            
            for row in data:
                values = []
                for col in available_cols:
                    val = str(row.get(col, ""))
                    # Clean up email format
                    if "@" in val:
                        val = val.split("@")[0]
                    # Truncate long values
                    if len(val) > 40:
                        val = val[:37] + "..."
                    values.append(val)
                md_table += "| " + " | ".join(values) + " |\n"
            
            return f"📋 **Database Results** (via local fallback):\n\n{md_table}\n\n_Found {len(data)} ticket(s)._"
        elif isinstance(data, list) and len(data) == 0:
            return "No tickets found matching your query."
        else:
            return f"📋 **Database Results** (via local fallback):\n\n{result}"
            
    except Exception as e:
        print(f"Direct database query failed: {e}")
        import traceback
        traceback.print_exc()
        return None


app = Flask(__name__)
hybrid_agent = None
adk_agent = None
conversation_state = {}
embedding_backfill_attempted = False


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

        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
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
                <div class="subtitle">Use these prompts in the chat to create, update, or search tickets.</div>
            </div>
            <a class="back-btn" href="/">← Back to Chat</a>
        </div>

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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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
            height: 100vh;
            overflow: hidden;
        }
        
        /* Animated Background */
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
        
        /* Main Layout */
        .app-container {
            display: flex;
            height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            padding: 24px;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 32px;
        }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        }
        
        .logo-text {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text) 0%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .logo-subtitle {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Status Cards */
        .status-section {
            margin-bottom: 24px;
        }
        
        .section-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }
        
        .status-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.3s ease;
        }
        
        .status-card:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--primary);
        }
        
        .status-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        .status-name {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 500;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 12px var(--success);
            animation: pulse 2s infinite;
        }
        
        .status-dot.offline {
            background: var(--danger);
            box-shadow: 0 0 12px var(--danger);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-badge {
            font-size: 10px;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-badge.local {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
        }
        
        .status-badge.cloud {
            background: rgba(6, 182, 212, 0.2);
            color: var(--accent);
        }
        
        .status-desc {
            font-size: 12px;
            color: var(--text-muted);
        }
        
        /* Quick Actions */
        .quick-actions {
            flex: 1;
        }
        
        .action-btn {
            width: 100%;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 12px 16px;
            color: var(--text);
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
            text-align: left;
        }
        
        .action-btn:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--primary);
            transform: translateX(4px);
        }
        
        .action-icon {
            font-size: 16px;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: auto;
            padding-top: 24px;
            border-top: 1px solid var(--glass-border);
        }
        
        .stat-item {
            text-align: center;
            padding: 12px;
            background: var(--glass);
            border-radius: 10px;
            border: 1px solid var(--glass-border);
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        
        /* Top Bar */
        .top-bar {
            padding: 20px 32px;
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .page-title {
            font-size: 24px;
            font-weight: 600;
        }
        
        .top-bar-actions {
            display: flex;
            gap: 12px;
        }
        
        .icon-btn {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.3s ease;
        }
        
        .icon-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        /* Chat Area */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 32px;
            scroll-behavior: smooth;
        }
        
        .chat-area::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-area::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-area::-webkit-scrollbar-thumb {
            background: var(--dark-lighter);
            border-radius: 3px;
        }
        
        /* Welcome Card */
        .welcome-card {
            max-width: 600px;
            margin: 60px auto;
            text-align: center;
            padding: 48px;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            backdrop-filter: blur(20px);
        }
        
        .welcome-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin: 0 auto 24px;
            box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
        }
        
        .welcome-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--text) 0%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .welcome-desc {
            color: var(--text-muted);
            font-size: 15px;
            line-height: 1.6;
            margin-bottom: 32px;
        }
        
        .feature-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            text-align: left;
        }
        
        .feature-item {
            padding: 16px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            border: 1px solid var(--glass-border);
        }
        
        .feature-item h4 {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .feature-item p {
            font-size: 12px;
            color: var(--text-muted);
        }
        
        /* Messages */
        .message {
            margin-bottom: 24px;
            display: flex;
            gap: 16px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        
        .message.user .message-avatar {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        }
        
        .message.assistant .message-avatar {
            background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
        }
        
        .message-content {
            max-width: 70%;
        }
        
        .message-bubble {
            padding: 16px 20px;
            border-radius: 16px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .message.user .message-bubble {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-bubble {
            background: var(--dark-light);
            border: 1px solid var(--glass-border);
            border-bottom-left-radius: 4px;
        }
        
        .message-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 8px;
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .model-tag {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 10px;
            font-weight: 600;
        }
        
        .model-tag.local {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
        }
        
        .model-tag.cloud {
            background: rgba(6, 182, 212, 0.2);
            color: var(--accent);
        }
        
        /* Table Container with Scrollbar */
        .table-container {
            max-height: 800px;
            overflow-y: auto;
            overflow-x: auto;
            margin: 16px 0;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.3);
        }
        
        .table-container::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        .table-container::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }
        
        .table-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border-radius: 4px;
        }
        
        .table-container::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }
        
        /* Table Styles */
        .message-bubble table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            border-radius: 12px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.3);
        }
        
        .message-bubble .table-container table {
            margin: 0;
            background: transparent;
        }
        
        .message-bubble th {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 14px 16px;
            text-align: left;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .message-bubble td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--glass-border);
            color: var(--text);
        }
        
        .message-bubble tr:last-child td {
            border-bottom: none;
        }
        
        .message-bubble tr:hover td {
            background: rgba(99, 102, 241, 0.1);
        }
        
        .message-bubble code {
            background: rgba(0, 0, 0, 0.4);
            padding: 3px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--accent);
        }
        
        .message-bubble pre {
            background: rgba(0, 0, 0, 0.5);
            padding: 16px;
            border-radius: 12px;
            overflow-x: auto;
            margin: 12px 0;
            border: 1px solid var(--glass-border);
        }
        
        .message-bubble pre code {
            background: none;
            padding: 0;
            color: var(--text);
        }
        
        .message-bubble strong {
            color: var(--accent);
            font-weight: 600;
        }
        
        .message-bubble a {
            color: var(--primary);
        }
        
        .message-bubble ul, .message-bubble ol {
            margin: 12px 0;
            padding-left: 24px;
        }
        
        .message-bubble li {
            margin: 6px 0;
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: none;
            padding: 20px;
            background: var(--dark-light);
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            width: fit-content;
        }
        
        .typing-indicator.active {
            display: block;
        }
        
        .typing-dots {
            display: flex;
            gap: 6px;
        }
        
        .typing-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary);
            animation: bounce 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-12px); }
        }
        
        /* Input Area */
        .input-area {
            padding: 24px 32px;
            background: var(--glass);
            backdrop-filter: blur(20px);
            border-top: 1px solid var(--glass-border);
        }
        
        .input-container {
            display: flex;
            gap: 16px;
            max-width: 100%;
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .input-field {
            width: 100%;
            padding: 16px 24px;
            background: var(--dark-light);
            border: 2px solid var(--dark-lighter);
            border-radius: 16px;
            color: var(--text);
            font-size: 15px;
            font-family: inherit;
            outline: none;
            transition: all 0.3s ease;
        }
        
        .input-field::placeholder {
            color: var(--text-muted);
        }
        
        .input-field:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
        }
        
        .send-btn {
            padding: 16px 32px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            border: none;
            border-radius: 16px;
            color: white;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        }
        
        .send-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.4);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .stop-btn {
            padding: 16px 24px;
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            border: none;
            border-radius: 16px;
            color: white;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            display: none;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
        }
        
        .stop-btn.visible {
            display: flex;
        }
        
        .stop-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(239, 68, 68, 0.4);
        }
        
        .stop-btn .stop-icon {
            width: 14px;
            height: 14px;
            background: white;
            border-radius: 2px;
        }

        .attach-btn {
            padding: 16px 18px;
            background: var(--dark-light);
            border: 2px solid var(--dark-lighter);
            border-radius: 16px;
            color: var(--text);
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .attach-btn:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.2);
        }

        .attach-indicator {
            margin-top: 8px;
            font-size: 12px;
            color: var(--accent);
            min-height: 16px;
        }

        .attach-preview {
            margin-top: 8px;
            max-width: 220px;
            max-height: 150px;
            border-radius: 10px;
            border: 1px solid var(--glass-border);
            display: none;
            object-fit: contain;
        }

        .attach-preview-wrap {
            margin-top: 8px;
            width: fit-content;
            position: relative;
            display: none;
        }

        .attach-remove-btn {
            position: absolute;
            top: -8px;
            right: -8px;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            border: 1px solid var(--glass-border);
            background: rgba(15, 23, 42, 0.95);
            color: #fff;
            font-size: 14px;
            line-height: 1;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .attach-remove-btn:hover {
            border-color: #f87171;
            color: #fca5a5;
        }
        
        /* Responsive */
        @media (max-width: 1024px) {
            .sidebar { display: none; }
            .feature-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    <div class="grid-overlay"></div>
    
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="logo">
                <div class="logo-icon">🔧</div>
                <div>
                    <div class="logo-text">Tandion</div>
                    <div class="logo-subtitle">Bug Assistant</div>
                </div>
            </div>
            
            <div class="status-section">
                <div class="section-title">System Status</div>
                <div class="status-card">
                    <div class="status-header">
                        <span class="status-name">
                            <span class="status-dot" id="localDot"></span>
                            Local Model
                        </span>
                        <span class="status-badge local">Gemma-2B</span>
                    </div>
                    <div class="status-desc">Fast • Free • Offline-capable</div>
                </div>
                <div class="status-card">
                    <div class="status-header">
                        <span class="status-name">
                            <span class="status-dot" id="cloudDot"></span>
                            Cloud Model
                        </span>
                        <span class="status-badge cloud">Gemini</span>
                    </div>
                    <div class="status-desc">Powerful • Tools • Database Access</div>
                </div>
            </div>
            
            <div class="quick-actions">
                <div class="section-title">Quick Actions</div>
                <button class="action-btn" onclick="window.location='/help'">
                    <span class="action-icon">💡</span>
                    Help & Examples
                </button>
                <button class="action-btn" onclick="quickAction('Show all open tickets')">
                    <span class="action-icon">📋</span>
                    Open Tickets
                </button>
                <button class="action-btn" onclick="quickAction('Show critical priority tickets')">
                    <span class="action-icon">🚨</span>
                    Critical Issues
                </button>
                <button class="action-btn" onclick="quickAction('Show high priority tickets')">
                    <span class="action-icon">⚠️</span>
                    High Priority
                </button>
                <button class="action-btn" onclick="quickAction('Show closed tickets')">
                    <span class="action-icon">✅</span>
                    Closed Tickets
                </button>
            </div>
            
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="queryCount">0</div>
                    <div class="stat-label">Queries</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="sessionTime">0m</div>
                    <div class="stat-label">Session</div>
                </div>
            </div>
        </aside>
        
        <!-- Main Content -->
        <main class="main-content">
            <div class="top-bar">
                <h1 class="page-title">🤖 AI Assistant</h1>
                <div class="top-bar-actions">
                    <button class="icon-btn" onclick="clearChat()" title="Clear Chat">🗑️</button>
                    <button class="icon-btn" onclick="refreshStatus()" title="Refresh Status">🔄</button>
                </div>
            </div>
            
            <div class="chat-area" id="chatArea">
                <div class="welcome-card" id="welcomeCard">
                    <div class="welcome-icon">🔧</div>
                    <h2 class="welcome-title">Welcome to Tandion Bug Assistant</h2>
                    <p class="welcome-desc">
                        Your AI-powered software bug management system. I can help you track issues, 
                        analyze bugs, and manage your tickets efficiently.
                    </p>
                    <div class="feature-grid">
                        <div class="feature-item">
                            <h4>🏠 Local Processing</h4>
                            <p>Fast responses using Gemma-2B for simple queries</p>
                        </div>
                        <div class="feature-item">
                            <h4>☁️ Cloud Power</h4>
                            <p>Complex analysis with Gemini + database tools</p>
                        </div>
                        <div class="feature-item">
                            <h4>📊 Smart Fallback</h4>
                            <p>Automatic switching when cloud is unavailable</p>
                        </div>
                        <div class="feature-item">
                            <h4>🔍 Ticket Search</h4>
                            <p>Query by status, priority, or assignee</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-container">
                    <div class="input-wrapper">
                        <input 
                            type="text" 
                            class="input-field" 
                            id="messageInput"
                            placeholder="Ask about tickets, bugs, or debugging..."
                            onkeypress="handleKeyPress(event)"
                        />
                        <input type="file" id="imageInput" accept="image/*" style="display:none" />
                        <div class="attach-indicator" id="attachIndicator"></div>
                        <div id="attachPreviewWrap" class="attach-preview-wrap">
                            <img id="attachPreview" class="attach-preview" alt="Image preview" />
                            <button type="button" class="attach-remove-btn" onclick="clearPendingImage()" title="Remove image">✕</button>
                        </div>
                    </div>
                    <button class="attach-btn" id="attachButton" onclick="document.getElementById('imageInput').click()" title="Attach Image">
                        📷
                    </button>
                    <button class="send-btn" id="sendButton" onclick="sendMessage()">
                        <span>Send</span>
                        <span>→</span>
                    </button>
                    <button class="stop-btn" id="stopButton" onclick="stopResponse()">
                        <div class="stop-icon"></div>
                        <span>Stop</span>
                    </button>
                </div>
            </div>
        </main>
    </div>
    
    <script>
        let conversationHistory = [];
        const chatStateKey = 'tandion_chat_state';
        const clientIdKey = 'tandion_client_id';
        let clientId = localStorage.getItem(clientIdKey);
        if (!clientId) {
            const fallbackId = 'client_' + Math.random().toString(36).slice(2);
            clientId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : fallbackId;
            localStorage.setItem(clientIdKey, clientId);
        }
        let queryCount = 0;
        let sessionStart = Date.now();
        let currentAbortController = null;
        let isGenerating = false;
        let pendingImageBase64 = null;
        let pendingImageMime = null;
        let pendingImageName = null;
        let pendingImagePreviewUrl = null;

        function isReloadNavigation() {
            const navEntries = performance.getEntriesByType && performance.getEntriesByType('navigation');
            if (navEntries && navEntries.length > 0) {
                return navEntries[0].type === 'reload';
            }
            return performance.navigation && performance.navigation.type === 1;
        }

        function saveChatState() {
            const payload = {
                conversationHistory,
                queryCount,
                sessionStart,
            };
            sessionStorage.setItem(chatStateKey, JSON.stringify(payload));
        }

        function loadChatState() {
            if (isReloadNavigation()) {
                sessionStorage.removeItem(chatStateKey);
                return;
            }

            const raw = sessionStorage.getItem(chatStateKey);
            if (!raw) return;

            try {
                const state = JSON.parse(raw);
                conversationHistory = Array.isArray(state.conversationHistory) ? state.conversationHistory : [];
                queryCount = Number.isFinite(state.queryCount) ? state.queryCount : 0;
                sessionStart = Number.isFinite(state.sessionStart) ? state.sessionStart : Date.now();

                document.getElementById('queryCount').textContent = queryCount;

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
                sessionStorage.removeItem(chatStateKey);
            }
        }
        
        // Update session time
        setInterval(() => {
            const mins = Math.floor((Date.now() - sessionStart) / 60000);
            document.getElementById('sessionTime').textContent = mins + 'm';
        }, 60000);
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }

        function updateAttachIndicator() {
            const indicator = document.getElementById('attachIndicator');
            const preview = document.getElementById('attachPreview');
            const previewWrap = document.getElementById('attachPreviewWrap');
            if (pendingImageName) {
                indicator.textContent = `Attached image: ${pendingImageName} (auto cloud)`;
                if (pendingImagePreviewUrl) {
                    preview.src = pendingImagePreviewUrl;
                    preview.style.display = 'block';
                    previewWrap.style.display = 'block';
                }
            } else {
                indicator.textContent = '';
                preview.src = '';
                preview.style.display = 'none';
                previewWrap.style.display = 'none';
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
                pendingImageName = `${labelPrefix}: ${file.name || 'clipboard-image.png'}`;
                updateAttachIndicator();
            };
            reader.readAsDataURL(file);
        }

        function setPendingImageFromDataUrl(dataUrl, label = 'Pasted image: clipboard-image.png') {
            if (!dataUrl || !dataUrl.startsWith('data:image/')) return false;
            const parts = dataUrl.split(',');
            if (parts.length < 2) return false;
            const mimeMatch = parts[0].match(/^data:(image\/[a-zA-Z0-9.+-]+);base64$/);
            pendingImageBase64 = parts[1] || null;
            pendingImageMime = (mimeMatch && mimeMatch[1]) ? mimeMatch[1] : 'image/png';
            pendingImagePreviewUrl = dataUrl;
            pendingImageName = label;
            updateAttachIndicator();
            return true;
        }

        function tryExtractImageFromClipboard(event) {
            const clipboard = event.clipboardData;
            if (!clipboard) return false;

            // 1) Files API path
            if (clipboard.files && clipboard.files.length > 0) {
                const file = clipboard.files[0];
                if (file && file.type && file.type.startsWith('image/')) {
                    setPendingImageFromFile(file, 'Pasted image');
                    return true;
                }
            }

            // 2) Clipboard items path
            if (clipboard.items) {
                for (const item of clipboard.items) {
                    if (item.type && item.type.startsWith('image/')) {
                        const file = item.getAsFile();
                        if (file) {
                            setPendingImageFromFile(file, 'Pasted image');
                            return true;
                        }
                    }
                }
            }

            // 3) HTML path with inline data URL image
            const html = clipboard.getData && clipboard.getData('text/html');
            if (html) {
                const dataUrlMatch = html.match(/src=["'](data:image\/[a-zA-Z0-9.+-]+;base64,[^"']+)["']/i);
                if (dataUrlMatch && setPendingImageFromDataUrl(dataUrlMatch[1])) {
                    return true;
                }
            }

            // 4) Plain text path containing data URL
            const text = clipboard.getData && clipboard.getData('text/plain');
            if (text && text.startsWith('data:image/')) {
                return setPendingImageFromDataUrl(text);
            }

            return false;
        }
        
        function quickAction(query) {
            document.getElementById('messageInput').value = query;
            sendMessage();
        }
        
        function clearChat() {
            const chatArea = document.getElementById('chatArea');
            chatArea.innerHTML = `
                <div class="welcome-card" id="welcomeCard">
                    <div class="welcome-icon">🔧</div>
                    <h2 class="welcome-title">Welcome to Tandion Bug Assistant</h2>
                    <p class="welcome-desc">
                        Your AI-powered software bug management system. Ready to help!
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
                    document.getElementById('localDot').className = 
                        'status-dot' + (data.local_available ? '' : ' offline');
                    document.getElementById('cloudDot').className = 
                        'status-dot' + (data.cloud_available ? '' : ' offline');
                });
        }
        
        function addMessage(text, isUser, modelUsed = null, imageUrl = null) {
            const chatArea = document.getElementById('chatArea');
            const welcome = document.getElementById('welcomeCard');
            if (welcome) welcome.remove();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = isUser ? '👤' : '🤖';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            if (isUser) {
                if (imageUrl) {
                    const img = document.createElement('img');
                    img.src = imageUrl;
                    img.alt = 'Uploaded image';
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
                marked.setOptions({ gfm: true, breaks: true });
                bubble.innerHTML = marked.parse(text);
                
                // Wrap tables in scrollable container
                const tables = bubble.querySelectorAll('table');
                tables.forEach(table => {
                    const container = document.createElement('div');
                    container.className = 'table-container';
                    table.parentNode.insertBefore(container, table);
                    container.appendChild(table);
                });
            }
            
            content.appendChild(bubble);
            
            if (!isUser && modelUsed) {
                const meta = document.createElement('div');
                meta.className = 'message-meta';
                const isLocal = modelUsed === 'gemma-2b-local';
                const isLocalTools = modelUsed === 'gemma-2b-tools';
                
                let modelLabel, modelIcon, tagClass;
                if (isLocalTools) {
                    modelLabel = 'Gemma-2B + Tools';
                    modelIcon = '🔧';
                    tagClass = 'local';
                } else if (isLocal) {
                    modelLabel = 'Gemma-2B';
                    modelIcon = '🏠';
                    tagClass = 'local';
                } else {
                    modelLabel = 'Gemini';
                    modelIcon = '☁️';
                    tagClass = 'cloud';
                }
                
                meta.innerHTML = `
                    <span class="model-tag ${tagClass}">
                        ${modelIcon} ${isLocal || isLocalTools ? 'Local' : 'Cloud'} • ${modelLabel}
                    </span>
                    <span>${new Date().toLocaleTimeString()}</span>
                `;
                content.appendChild(meta);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendButton');
            const chatArea = document.getElementById('chatArea');
            
            const message = input.value.trim();
            if (!message && !pendingImageBase64) return;

            const imagePayloadBase64 = pendingImageBase64;
            const imagePayloadMime = pendingImageMime;
            const imagePreviewUrl = pendingImagePreviewUrl;

            const userMessageText = message;
            const historyUserText = message || 'Uploaded image';
            
            addMessage(userMessageText, true, null, imagePreviewUrl);
            input.value = '';
            // Clear attachment from composer immediately after send.
            clearPendingImage();
            sendBtn.disabled = true;
            
            // Show stop button
            const stopBtn = document.getElementById('stopButton');
            stopBtn.classList.add('visible');
            isGenerating = true;
            
            // Create abort controller for this request
            currentAbortController = new AbortController();
            
            // Show typing indicator
            const typing = document.createElement('div');
            typing.className = 'typing-indicator active';
            typing.innerHTML = '<div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
            chatArea.appendChild(typing);
            chatArea.scrollTop = chatArea.scrollHeight;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message,
                        history: conversationHistory,
                        client_id: clientId,
                        image_data: imagePayloadBase64,
                        image_mime: imagePayloadMime
                    }),
                    signal: currentAbortController.signal
                });
                
                const data = await response.json();
                typing.remove();
                
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
            } catch (error) {
                typing.remove();
                if (error.name === 'AbortError') {
                    addMessage('⏹️ Response stopped by user.', false);
                } else {
                    addMessage('Connection error. Please try again.', false);
                }
            }
            
            // Hide stop button and reset state
            stopBtn.classList.remove('visible');
            isGenerating = false;
            currentAbortController = null;
            sendBtn.disabled = false;
            input.focus();
        }
        
        function stopResponse() {
            if (currentAbortController && isGenerating) {
                currentAbortController.abort();
                isGenerating = false;
                document.getElementById('stopButton').classList.remove('visible');
            }
        }
        
        // Initial status check
        document.getElementById('imageInput').addEventListener('change', event => {
            const file = event.target.files && event.target.files[0];
            if (!file) return;
            setPendingImageFromFile(file, 'Attached image');
        });

        document.getElementById('messageInput').addEventListener('paste', event => {
            if (tryExtractImageFromClipboard(event)) {
                event.preventDefault();
            }
        });

        // Support paste when focus is outside the input (common with screenshots / copy as image).
        document.addEventListener('paste', event => {
            const target = event.target;
            if (target && target.id === 'messageInput') return;
            if (tryExtractImageFromClipboard(event)) {
                event.preventDefault();
            }
        });

        loadChatState();
        refreshStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/help')
def help_page():
    """Serve the help/examples page"""
    return render_template_string(HELP_TEMPLATE)


def init_agents():
    """Initialize both hybrid agent and ADK agent"""
    global hybrid_agent, adk_agent, embedding_backfill_attempted
    
    # Initialize hybrid agent (for local Gemma)
    if hybrid_agent is None:
        try:
            hybrid_agent = HybridAgent(enable_local=True)
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
    global hybrid_agent, adk_agent
    
    init_agents()
    
    local_available = (
        hybrid_agent is not None and
        hybrid_agent.local_model is not None and 
        hybrid_agent.local_model.is_available()
    )
    
    cloud_available = adk_agent is not None
    
    return jsonify({
        "local_available": local_available,
        "cloud_available": cloud_available
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and route to appropriate model"""
    global hybrid_agent, adk_agent
    
    # Initialize agents if needed
    init_agents()
    
    if hybrid_agent is None:
        return jsonify({
            "error": "Failed to initialize hybrid agent"
        }), 500
    
    data = request.json
    message = data.get('message', '').strip()
    history = data.get('history', [])
    client_id = data.get('client_id') or request.remote_addr or "anonymous"
    image_data = data.get('image_data')
    image_mime = data.get('image_mime', 'image/png')
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
        # Accept forms like: @cloud question, @cloud: question, @cloud
        message = re.sub(r'^@cloud\s*:?', '', message, flags=re.IGNORECASE).strip()
        if not message:
            return jsonify({
                "response": "Please add a query after @cloud, for example: @cloud show open tickets",
                "model_used": "gemini-cloud",
                "classification": "FORCED_CLOUD",
                "error": True,
            })
        print(f"🌩️ Force Cloud Gemini requested for: {message}")

    if image_bytes is not None:
        force_cloud = True
        if not message:
            message = "Analyze the uploaded image and provide troubleshooting steps."

    state = _get_client_state(client_id)
    message_lower = message.lower()

    if force_cloud and state["stage"] == "awaiting_fix_confirmation":
        # Explicit cloud request should not be blocked by yes/no confirmation gate.
        state["stage"] = "idle"
        state["last_issue"] = ""

    if state["stage"] == "awaiting_fix_confirmation":
        if _is_affirmative(message_lower):
            state["stage"] = "idle"
            state["last_issue"] = ""
            return jsonify({
                "response": "✅ Glad it is fixed. I will not create a ticket. Anything else you need?",
                "model_used": "triage",
                "classification": "FIX_CONFIRMED"
            })
        if _is_negative(message_lower):
            issue_text = _derive_issue_text_for_ticket(state.get("last_issue", ""), history)
            state["stage"] = "idle"
            state["last_issue"] = ""
            if not issue_text:
                return jsonify({
                    "response": "I can create a ticket, but I need the issue details first. What problem should I file?",
                    "model_used": "triage",
                    "classification": "NEED_DETAILS"
                })
            ticket_creator = get_ticket_creator()
            ticket_response = ticket_creator.create_ticket_from_query(issue_text)
            return jsonify({
                "response": ticket_response,
                "model_used": "database-direct",
                "classification": "TICKET_CREATED"
            })

        if _looks_like_new_query(message_lower):
            # User started a new request; do not keep blocking with yes/no gate.
            state["stage"] = "idle"
            state["last_issue"] = ""
        else:
            return jsonify({
                "response": "Please confirm if the issue is fixed. Reply 'yes' or 'no'.",
                "model_used": "triage",
                "classification": "NEED_CONFIRMATION"
            })

    if state["stage"] == "awaiting_problem":
        state["stage"] = "idle"
        message_lower = message.lower()
        state["last_issue"] = message
    
    # Build context from conversation history
    context = {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversation_history": [
            f"{msg['role']}: {msg['content']}" 
            for msg in history[-5:]  # Last 5 messages
        ] if history else []
    }

    if _needs_problem_details(message_lower):
        state["stage"] = "awaiting_problem"
        return jsonify({
            "response": "What problem should I fix? Please describe the issue (symptoms, error text, when it happens).",
            "model_used": "triage",
            "classification": "NEED_DETAILS"
        })

    if not _is_ticket_query(message_lower):
        issue_text = state.get("last_issue") or message
        state["last_issue"] = issue_text
        ask_fix_confirmation = _is_troubleshoot_request(message_lower)

        if not force_cloud and local_llm_with_tools and local_llm_with_tools.is_available():
            try:
                local_tool_response = local_llm_with_tools.chat(issue_text)
                response_text = local_tool_response or ""
                response_text = response_text.strip() or "I can help, but I need more details about the issue."
                if ask_fix_confirmation:
                    response_text += "\n\n---\n\nDid that fix it? Reply **yes** or **no**. If no, I will create a ticket."
                    state["stage"] = "awaiting_fix_confirmation"
                return jsonify({
                    "response": response_text,
                    "model_used": "gemma-2b-tools",
                    "classification": "TRIAGE"
                })
            except Exception as local_err:
                print(f"Local LLM with tools failed: {local_err}, trying cloud...")

        if adk_agent is None:
            if force_cloud:
                return jsonify({
                    "response": "@cloud was requested, but Cloud Gemini is not available right now. Check API key/network and try again.",
                    "model_used": "gemini-cloud",
                    "classification": "FORCED_CLOUD",
                    "error": True,
                })
            if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
                try:
                    local_response = hybrid_agent.local_model.generate(issue_text)
                    response_text = local_response or ""
                except Exception:
                    response_text = "I need more details about the issue to help you."
            else:
                response_text = "I'm currently offline. Please share the issue details and try again later."

            google = search_google(issue_text)
            stack = query_stackexchange(issue_text)
            google_section = _format_search_section("🔍 Web Search Results", google.get("results", []))
            stack_section = _format_search_section("📚 StackOverflow Related Issues", stack.get("results", []))
            extra_sections = "\n\n".join([s for s in [google_section, stack_section] if s])
            if extra_sections:
                response_text += "\n\n---\n\n" + extra_sections

            if ask_fix_confirmation:
                response_text += "\n\n---\n\nDid that fix it? Reply **yes** or **no**. If no, I will create a ticket."
                state["stage"] = "awaiting_fix_confirmation"
            return jsonify({
                "response": response_text,
                "model_used": "offline",
                "classification": "TRIAGE"
            })

        try:
            import asyncio
            import uuid
            import traceback
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
                # If image processing fails, retry with text-only so user still gets help.
                if image_bytes is not None:
                    print(f"Cloud triage image processing failed, retrying text-only: {image_run_err}")
                    response_text = asyncio.run(run_agent_async(include_image=False))
                    response_text = (
                        "I couldn't fully process the image this time, so I used your text prompt instead.\n\n"
                        + (response_text or "")
                    )
                else:
                    raise

            response_text = response_text or "I can help, but I need more details about the issue."

            google = search_google(issue_text)
            stack = query_stackexchange(issue_text)
            google_section = _format_search_section("🔍 Web Search Results", google.get("results", []))
            stack_section = _format_search_section("📚 StackOverflow Related Issues", stack.get("results", []))
            extra_sections = "\n\n".join([s for s in [google_section, stack_section] if s])
            if extra_sections:
                response_text += "\n\n---\n\n" + extra_sections

            if ask_fix_confirmation:
                response_text += "\n\n---\n\nDid that fix it? Reply **yes** or **no**. If no, I will create a ticket."
                state["stage"] = "awaiting_fix_confirmation"
            return jsonify({
                "response": response_text,
                "model_used": "gemini-cloud",
                "classification": "TRIAGE"
            })
        except Exception as e:
            print(f"Cloud triage error: {e}")
            traceback.print_exc()
            response_text = (
                "Cloud analysis failed for this request. I can still help if you share the exact error text "
                "or retry with @cloud and a smaller/clearer image."
            )
            if ask_fix_confirmation:
                response_text += "\n\n---\n\nDid that fix it? Reply **yes** or **no**. If no, I will create a ticket."
                state["stage"] = "awaiting_fix_confirmation"
            return jsonify({
                "response": response_text,
                "model_used": "error",
                "classification": "TRIAGE"
            })
    
    # Process query through hybrid agent
    try:
        result = hybrid_agent.process_query(message, context=context)
        
        # If @cloud prefix used, skip to cloud processing
        if force_cloud:
            result["handled_locally"] = False
            result["classification"] = "FORCED_CLOUD"
        
        if result["handled_locally"]:
            return jsonify({
                "response": result["response"],
                "model_used": result["model_used"],
                "classification": result["classification"]
            })
        else:
            # If force_cloud, skip local LLM and go straight to Cloud Gemini
            if not force_cloud:
                # For complex queries, try local LLM with tools FIRST (to avoid quota issues)
                # Only fall back to cloud Gemini if local tools can't handle it
            
                # Check if local LLM with tools can handle this query
                if local_llm_with_tools and local_llm_with_tools.is_available():
                    try:
                        print(f"Processing query with Local LLM + Tools: {message}")
                        local_tool_response = local_llm_with_tools.chat(message)
                        if local_tool_response and local_tool_response.strip():
                            return jsonify({
                                "response": local_tool_response,
                                "model_used": "gemma-2b-tools",
                                "classification": result["classification"]
                            })
                    except Exception as local_err:
                        print(f"Local LLM with tools failed: {local_err}, trying cloud...")
                
                # Try direct database query for ticket-related queries
                message_lower = message.lower()
                if any(kw in message_lower for kw in ["ticket", "open", "closed", "priority", "critical", "high", "status"]):
                    db_result = query_database_directly(message)
                    if db_result:
                        return jsonify({
                            "response": db_result,
                            "model_used": "database-direct",
                            "classification": result["classification"]
                        })
            
            # Use Cloud Gemini (either forced or as fallback)
            if adk_agent is None:
                if force_cloud:
                    return jsonify({
                        "response": "@cloud was requested, but Cloud Gemini is not available right now. Check API key/network and try again.",
                        "model_used": "gemini-cloud",
                        "classification": result["classification"],
                        "error": True,
                    })
                # No cloud agent available, use local model for general response
                if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
                    try:
                        local_response = hybrid_agent.local_model.generate(message)
                        return jsonify({
                            "response": local_response,
                            "model_used": "gemma-2b-local",
                            "classification": result["classification"]
                        })
                    except Exception:
                        pass
                
                return jsonify({
                    "response": "I'm currently running in offline mode. I can help with simple questions and database queries using the Quick Actions buttons.",
                    "model_used": "offline",
                    "classification": result["classification"],
                    "error": True
                })
            
            try:
                # Run the query through ADK agent (Gemini + tools)
                print(f"Processing complex query with Gemini + Tools: {message}")
                
                # ADK agents need to be invoked using a Runner with Content/Part objects
                import asyncio
                import uuid
                from google.adk.runners import InMemoryRunner
                from google.genai import types
                
                async def run_agent_async():
                    """Run the ADK agent using Runner with proper Content format"""
                    try:
                        # Create Content object with text + optional image parts
                        parts = [types.Part(text=message)]
                        if image_bytes is not None:
                            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
                        content = types.Content(role="user", parts=parts)

                        # Create runner - suppress the app name mismatch warning
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*mismatch.*")
                            runner = InMemoryRunner(agent=adk_agent, app_name="software_bug_assistant")

                        # Use unique session per request
                        user_id = "web_ui_user"
                        session_id = f"session_{uuid.uuid4().hex[:8]}"

                        # Create session first
                        await runner.session_service.create_session(
                            app_name="software_bug_assistant",
                            user_id=user_id,
                            session_id=session_id,
                        )

                        # Run the agent asynchronously - fail fast on quota errors
                        response_chunks = []
                        
                        try:
                            async for event in runner.run_async(
                                user_id=user_id,
                                session_id=session_id,
                                new_message=content
                            ):
                                # Extract text from event
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
                        except Exception as run_err:
                            error_str = str(run_err)
                            # Check for quota/overload errors - fail immediately to trigger local fallback
                            if any(x in error_str for x in ["429", "503", "RESOURCE_EXHAUSTED", "quota", "overloaded", "UNAVAILABLE"]):
                                print(f"⚠ Gemini API quota/overload error - switching to local fallback")
                                raise  # Let outer handler use local fallback
                            raise

                        return ''.join(response_chunks) if response_chunks else None
                    except Exception as e:
                        print(f"Runner.run_async failed: {e}")
                        raise  # Re-raise to trigger local fallback
                
                # Run the async function
                try:
                    response_text = asyncio.run(run_agent_async())
                except Exception as async_err:
                    # Propagate the error to trigger local fallback
                    raise async_err
                
                if response_text is None:
                    raise RuntimeError("No response from Gemini - trying local fallback")
                
                # Convert response to string
                # ADK agents typically return AgentResponse or similar
                response_str = None
                
                if hasattr(response_text, 'text'):
                    response_str = response_text.text
                elif hasattr(response_text, 'content'):
                    response_str = str(response_text.content)
                elif hasattr(response_text, 'output'):
                    response_str = str(response_text.output)
                elif hasattr(response_text, 'message'):
                    response_str = str(response_text.message)
                elif isinstance(response_text, dict):
                    # If it's a dict, extract text/content
                    response_str = response_text.get('text') or response_text.get('content') or response_text.get('output') or str(response_text)
                elif isinstance(response_text, str):
                    response_str = response_text
                else:
                    response_str = str(response_text)
                
                # Clean up the response
                if not response_str or response_str.strip() == "":
                    response_str = "I processed your query, but didn't receive a response. Please try rephrasing your question."
                
                return jsonify({
                    "response": response_str,
                    "model_used": "gemini-cloud",
                    "classification": result["classification"]
                })
                
            except Exception as e:
                # Fallback if ADK agent fails
                error_msg = str(e)
                print(f"ADK agent error: {error_msg}")
                
                # Check if it's an overload/quota error - fall back to direct DB query or local model
                is_quota_error = (
                    "503" in error_msg or 
                    "429" in error_msg or 
                    "overloaded" in error_msg.lower() or 
                    "UNAVAILABLE" in error_msg or
                    "RESOURCE_EXHAUSTED" in error_msg or
                    "quota" in error_msg.lower()
                )
                
                if is_quota_error:
                    print("Cloud Gemini quota/overload error, attempting local fallback...")
                    
                    # FIRST: Try LOCAL LLM WITH TOOL CALLING (Option B)
                    if local_llm_with_tools and local_llm_with_tools.is_available():
                        try:
                            print("  → Using Local LLM with Tool Calling...")
                            local_tool_response = local_llm_with_tools.chat(message)
                            if local_tool_response and local_tool_response.strip():
                                fallback_note = "🏠 _Cloud Gemini quota exceeded. Using **Local AI with Database Tools**._\n\n---\n\n"
                                return jsonify({
                                    "response": fallback_note + local_tool_response,
                                    "model_used": "gemma-2b-tools",
                                    "classification": result["classification"],
                                    "fallback": True,
                                    "local_tools": True
                                })
                        except Exception as local_tool_err:
                            print(f"  ⚠ Local LLM with tools failed: {local_tool_err}")
                    
                    # SECOND: Try direct database query if this looks like a DB query
                    db_result = query_database_directly(message)
                    if db_result:
                        print("✓ Direct database query successful!")
                        fallback_note = "⚠️ _Cloud Gemini quota exceeded, queried database directly._\n\n"
                        return jsonify({
                            "response": fallback_note + db_result,
                            "model_used": "gemma-2b-local",
                            "classification": result["classification"],
                            "fallback": True,
                            "direct_db": True
                        })
                    
                    # THIRD: Try local model for non-DB queries
                    if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
                        try:
                            local_response = hybrid_agent.local_model.generate(message)
                            if local_response and local_response.strip():
                                fallback_note = "⚠️ _Cloud Gemini quota exceeded, using local Gemma-2B._\n\n---\n\n"
                                return jsonify({
                                    "response": fallback_note + local_response,
                                    "model_used": "gemma-2b-local",
                                    "classification": result["classification"],
                                    "fallback": True
                                })
                        except Exception as local_err:
                            print(f"Local fallback also failed: {local_err}")
                    
                    # If all fallbacks failed, show friendly message
                    fallback_msg = """⚠️ **Google Gemini API quota exceeded.**

Your free tier daily limit has been reached. Don't worry - here's what's happening:

**Why this happens:**
- Google's free Gemini API has daily request limits
- You've hit the limit for today

**Solutions:**
1. ✅ **Use the local model** - Simple queries work offline with Gemma-2B
2. ⏰ **Wait until tomorrow** - Quota resets daily
3. 💳 **Upgrade API plan** - Get higher limits at [Google AI Studio](https://aistudio.google.com)

**What still works right now:**
- 🏠 Simple questions (greetings, basic math, definitions)
- 📊 Direct database queries via Quick Actions buttons
- 🔧 Local troubleshooting assistance

Try clicking the **Quick Actions** buttons on the left sidebar!"""
                
                elif "credentials" in error_msg.lower() or "auth" in error_msg.lower():
                    fallback_msg = "Cloud Gemini requires API credentials. Please check your .env file has GOOGLE_API_KEY set."
                elif "toolbox" in error_msg.lower() or "mcp" in error_msg.lower() or "5000" in error_msg:
                    fallback_msg = "Cloud Gemini needs MCP Toolbox server running. Please start it:\ncd deployment/mcp-toolbox\n./toolbox.exe --tools-file=tools.yaml"
                else:
                    # For other errors, try local LLM with tools first, then direct DB, then simple local model
                    print("Cloud error, attempting local fallback...")
                    
                    # Try LOCAL LLM WITH TOOL CALLING first (Option B)
                    if local_llm_with_tools and local_llm_with_tools.is_available():
                        try:
                            print("  → Using Local LLM with Tool Calling...")
                            local_tool_response = local_llm_with_tools.chat(message)
                            if local_tool_response and local_tool_response.strip():
                                fallback_note = "🏠 _Cloud processing encountered an issue. Using **Local AI with Database Tools**._\n\n---\n\n"
                                return jsonify({
                                    "response": fallback_note + local_tool_response,
                                    "model_used": "gemma-2b-tools",
                                    "classification": result["classification"],
                                    "fallback": True,
                                    "local_tools": True
                                })
                        except Exception as local_tool_err:
                            print(f"  ⚠ Local LLM with tools failed: {local_tool_err}")
                    
                    # Try direct database query 
                    db_result = query_database_directly(message)
                    if db_result:
                        print("✓ Direct database query successful!")
                        fallback_note = "⚠️ _Cloud processing had an issue, so I queried the database directly._\n\n"
                        return jsonify({
                            "response": fallback_note + db_result,
                            "model_used": "gemma-2b-local",
                            "classification": result["classification"],
                            "fallback": True,
                            "direct_db": True
                        })
                    
                    if hybrid_agent and hybrid_agent.local_model and hybrid_agent.local_model.is_available():
                        try:
                            local_response = hybrid_agent.local_model.generate(message)
                            if local_response and local_response.strip():
                                fallback_note = "⚠️ _Cloud processing encountered an issue, using local Gemma-2B instead._\n\n---\n\n"
                                return jsonify({
                                    "response": fallback_note + local_response,
                                    "model_used": "gemma-2b-local",
                                    "classification": result["classification"],
                                    "fallback": True
                                })
                        except Exception as local_err:
                            print(f"Local fallback also failed: {local_err}")
                    
                    fallback_msg = f"Cloud Gemini processing failed: {error_msg}\n\nPlease check:\n1. MCP Toolbox is running on port 5000\n2. .env has GOOGLE_API_KEY\n3. Database is accessible"
                
                return jsonify({
                    "response": fallback_msg,
                    "model_used": "gemini-cloud",
                    "classification": result["classification"],
                    "error": True
                })
    
    except Exception as e:
        return jsonify({
            "error": f"Error processing query: {str(e)}"
        }), 500


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
        print("⚠ Cloud Gemini: NOT AVAILABLE (check MCP Toolbox and .env)")
    
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

