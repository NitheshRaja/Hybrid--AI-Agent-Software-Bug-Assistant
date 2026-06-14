# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Local Tool Calling System for Gemma-2B
Implements function calling capabilities similar to Gemini Cloud.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()

# optional HTTP client
try:
    import requests
except Exception:
    requests = None

# embedding model cache
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the local embedding model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(model_name)
        return _embedding_model
    except Exception as e:
        print(f"⚠ Embedding model not available: {e}")
        _embedding_model = None
        return None


def _vector_to_pgvector(values: List[float]) -> str:
    """Format embedding values for pgvector input."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def _embed_text(text: str) -> Optional[str]:
    """Create a pgvector-formatted embedding for text."""
    model = _get_embedding_model()
    if model is None:
        return None

    try:
        embedding = model.encode(text, normalize_embeddings=True)
        return _vector_to_pgvector(embedding.tolist())
    except Exception as e:
        print(f"⚠ Failed to embed text: {e}")
        return None


def backfill_missing_embeddings(limit: int = 500) -> int:
    """Backfill embeddings for tickets missing vectors. Returns count updated."""
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")

    if not all([db_host, db_port, db_name, db_user, db_pass]):
        print("⚠ DB_* env vars missing; skipping embedding backfill")
        return 0

    model = _get_embedding_model()
    if model is None:
        return 0

    try:
        import psycopg

        updated = 0
        with psycopg.connect(
            host=db_host,
            port=int(db_port),
            dbname=db_name,
            user=db_user,
            password=db_pass,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ticket_id, title, description FROM tickets WHERE embedding IS NULL LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
                for ticket_id, title, description in rows:
                    text = f"{title}\n{description or ''}".strip()
                    embedding = model.encode(text, normalize_embeddings=True)
                    vector = _vector_to_pgvector(embedding.tolist())
                    cur.execute(
                        "UPDATE tickets SET embedding = %s::vector WHERE ticket_id = %s",
                        (vector, ticket_id),
                    )
                    updated += 1
        if updated:
            print(f"✓ Backfilled embeddings: {updated}")
        return updated
    except Exception as e:
        print(f"⚠ Embedding backfill failed: {e}")
        return 0

# ============================================================================
# Tool Definition Classes
# ============================================================================

@dataclass
class ToolParameter:
    """Defines a parameter for a tool"""
    name: str
    type: str  # "string", "integer", "boolean", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass
class Tool:
    """Defines a tool that can be called by the LLM"""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    category: str = "general"
    
    def get_schema(self) -> dict:
        """Get JSON schema for the tool"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM"""
    tool_name: str
    arguments: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "arguments": self.arguments
        }


@dataclass
class ToolResult:
    """Result from executing a tool"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error
        }


from .db_tools import (
    get_tickets_by_status,
    get_tickets_by_priority,
    get_ticket_by_id,
    get_tickets_by_assignee,
    update_ticket_status,
    update_ticket_priority,
    get_tickets_by_date_range,
    search_tickets_vector,
    update_ticket_embedding,
    db_create_ticket
)

def search_tickets(query: str) -> dict:
    """Search tickets using vector similarity"""
    try:
        embedding = _embed_text(query)
        if embedding is None:
            return {"error": "Embedding model not available", "tickets": []}
        return search_tickets_vector(embedding)
    except Exception as e:
        return {"error": str(e), "tickets": []}


# ============================================================================
# Simple Ticket Creation - Minimal questions, auto-assign
# ============================================================================

# Auto-assign based on priority
PRIORITY_ASSIGNEES = {
    'P0 - Critical': 'senior.engineer@tandion.com',
    'P1 - High': 'lead.developer@tandion.com',
    'P2 - Medium': 'support.engineer@tandion.com',
    'P3 - Low': 'helpdesk@tandion.com',
}


class SimpleTicketCreator:
    """Simple ticket creation - asks only if info is missing"""
    
    def __init__(self):
        self.waiting_for_details = False
        self.partial_info = ""
    
    def reset(self):
        self.waiting_for_details = False
        self.partial_info = ""
    
    def is_waiting(self) -> bool:
        return self.waiting_for_details
    
    def process_response(self, user_input: str) -> str:
        """User provided the missing details"""
        if not self.waiting_for_details or user_input.strip() == "":
            return None
        
        user_input = user_input.strip()
        
        if user_input.lower() in ['cancel', 'no', 'stop', 'nevermind']:
            self.reset()
            return "❌ Ticket creation cancelled."
        
        # Combine with any partial info we had
        full_text = f"{self.partial_info} {user_input}".strip() if self.partial_info else user_input
        self.reset()
        
        # Now create the ticket
        return self._create_ticket_now(full_text)
    
    def create_ticket_from_query(self, query: str) -> str:
        """Create ticket from user's message"""
        self.reset()
        
        # Extract the issue description from query
        issue_text = self._extract_issue_text(query)
        print(f"  📋 Extracted issue text: '{issue_text}' (len={len(issue_text)})")
        
        # If we have enough info (at least 5 chars), create immediately
        if len(issue_text) >= 5:
            print(f"  ✓ Enough info, creating ticket now...")
            return self._create_ticket_now(issue_text)
        
        # Not enough info - ask for details
        self.waiting_for_details = True
        self.partial_info = issue_text
        print(f"  ⚠ Not enough info, waiting for details...")
        return "📝 **What issue do you want to report?**\n\n_Describe your problem (e.g., \"VPN not connecting after update\")_"
    
    def _extract_issue_text(self, query: str) -> str:
        """Extract the actual issue from the query"""
        text = query
        
        # Remove common prefixes
        prefixes = [
            'create ticket', 'create a ticket', 'new ticket', 'add ticket',
            'open ticket', 'report bug', 'report issue', 'log ticket',
            'please create', 'can you create', 'i want to create',
            'because', 'for', 'about', 'regarding', ':'
        ]
        
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                text_lower = text.lower()
        
        return text.lstrip(':- ').strip()

    def _is_generic_issue_text(self, text: str) -> bool:
        cleaned = (text or '').strip().lower()
        if not cleaned:
            return True
        generic_patterns = [
            r"^how to fix (this|the) issue\??$",
            r"^fix (this|the) issue\??$",
            r"^this issue\??$",
            r"^issue\??$",
            r"^problem\??$",
            r"^help\??$",
        ]
        return any(re.match(p, cleaned) for p in generic_patterns)

    def _normalize_issue_text(self, text: str) -> str:
        """Normalize noisy user text into a ticket-ready issue summary."""
        cleaned = re.sub(r"\s+", " ", (text or "")).strip().lstrip(':- ').strip()
        if not cleaned:
            return ""

        # If user pasted generic phrasing with a trailing specific detail, keep the detail.
        m = re.search(r"(?:issue|problem)\s*[:\-]\s*(.+)$", cleaned, re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()

        return cleaned

    def _build_title(self, issue_text: str) -> str:
        """Create a concise, informative title from issue details."""
        normalized = self._normalize_issue_text(issue_text)
        if not normalized:
            return "General issue reported"

        stop_code_match = re.search(r"\b([A-Z][A-Z0-9_]{4,})\b", normalized.upper())
        lower = normalized.lower()
        if ("blue screen" in lower or "bsod" in lower or "stop code" in lower) and stop_code_match:
            return f"BSOD: {stop_code_match.group(1)}"[:100]

        first_sentence = re.split(r"[\n.!?]+", normalized)[0].strip()
        candidate = first_sentence if first_sentence else normalized
        if self._is_generic_issue_text(candidate):
            candidate = f"Issue reported: {normalized}"
        return candidate[:100]
    
    def _detect_priority(self, text: str) -> str:
        """Detect priority from text"""
        text_lower = text.lower()

        critical_signals = [
            'critical', 'urgent', 'emergency', 'down', 'crashed', 'crash',
            'bsod', 'blue screen', 'stop code', 'kernel panic', 'reboot loop',
            'manually_initiated_crash'
        ]
        if any(w in text_lower for w in critical_signals):
            return 'P0 - Critical'
        elif any(w in text_lower for w in ['important', 'asap', 'serious', 'major']):
            return 'P1 - High'
        elif any(w in text_lower for w in ['slow', 'lag', 'minor', 'small']):
            return 'P3 - Low'
        else:
            return 'P2 - Medium'  # Default
    
    def _create_ticket_now(self, issue_text: str) -> str:
        """Create the ticket immediately"""
        print(f"  📝 Creating ticket with text: '{issue_text}'")

        normalized_issue = self._normalize_issue_text(issue_text)
        if self._is_generic_issue_text(normalized_issue):
            return "❌ **Error:** Issue details are too generic. Please provide the error message or symptoms so I can create a useful ticket."

        title = self._build_title(normalized_issue)
        description = normalized_issue
        
        # Auto-detect priority from text
        priority = self._detect_priority(issue_text)
        print(f"  📊 Detected priority: {priority}")
        
        # Auto-assign based on priority
        assignee = PRIORITY_ASSIGNEES.get(priority, 'helpdesk@tandion.com')
        print(f"  👤 Auto-assigned to: {assignee}")
        
        # Create ticket in DB
        result = create_ticket(
            title=title,
            description=description,
            assignee=assignee,
            priority=priority,
            status='Open'
        )
        
        # Check if this is a duplicate
        if result.get('duplicate'):
            existing_ticket = result.get('existing_ticket', {})
            existing_id = result.get('ticket_id', 'N/A')
            existing_priority = existing_ticket.get('priority', 'Unknown')
            existing_assignee = existing_ticket.get('assignee', 'Unknown')
            
            return f"""⚠️ **Duplicate Ticket Detected!**

A ticket with the same title already exists:

| Field | Value |
|-------|-------|
| **Existing ID** | {existing_id} |
| **Title** | {title} |
| **Priority** | {existing_priority} |
| **Assigned To** | {existing_assignee.split('@')[0] if '@' in str(existing_assignee) else existing_assignee} |
| **Status** | {existing_ticket.get('status', 'Unknown')} |

_Please check this existing ticket before creating a new one._"""
        
        if result.get('success'):
            ticket_id = _extract_ticket_id(result.get('ticket_id')) or 'N/A'
            return f"""✅ **Ticket Created!**

| Field | Value |
|-------|-------|
| **ID** | {ticket_id} |
| **Title** | {title} |
| **Priority** | {priority} |
| **Assigned To** | {assignee.split('@')[0]} |
| **Status** | Open |

_Your ticket has been saved and assigned._"""
        else:
            return f"❌ **Error:** {result.get('error', 'Could not create ticket')}"


# Global instance
_ticket_creator: Optional[SimpleTicketCreator] = None

def get_ticket_creator() -> SimpleTicketCreator:
    global _ticket_creator
    if _ticket_creator is None:
        _ticket_creator = SimpleTicketCreator()
    return _ticket_creator


def _is_similar_title(title1: str, title2: str, threshold: float = 0.75) -> bool:
    """Check if two titles are similar enough to be considered duplicates.
    Uses word-level overlap (Jaccard similarity) for fuzzy matching.
    """
    t1 = title1.lower().strip()
    t2 = title2.lower().strip()
    
    # Exact match
    if t1 == t2:
        return True
    
    # One title contains the other
    if t1 in t2 or t2 in t1:
        return True
    
    # Word-level Jaccard similarity
    words1 = set(t1.split())
    words2 = set(t2.split())
    
    # Remove very common stop words
    stop_words = {'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'not'}
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return False
    
    intersection = words1 & words2
    union = words1 | words2
    similarity = len(intersection) / len(union) if union else 0
    
    return similarity >= threshold


def create_ticket(title: str, description: str, assignee: str = "", priority: str = "P3 - Low", status: str = "Open") -> dict:
    """Create a new ticket"""
    try:
        # Check for duplicate tickets across all active statuses (Open, In Progress)
        all_active_tickets = []
        for check_status in ["Open", "In Progress"]:
            status_result = get_tickets_by_status(check_status)
            if status_result.get('success') and status_result.get('tickets'):
                all_active_tickets.extend(status_result['tickets'])
        
        if all_active_tickets:
            title_lower = title.lower().strip()
            
            for ticket in all_active_tickets:
                existing_title = ticket.get('title', '').lower().strip()
                # Check for exact match or similar titles using fuzzy matching
                if _is_similar_title(title_lower, existing_title):
                    return {
                        "success": False,
                        "duplicate": True,
                        "ticket_id": ticket.get('ticket_id'),
                        "existing_ticket": ticket,
                        "message": f"A ticket with the same or similar title already exists (ID: {ticket.get('ticket_id')}). Please check existing tickets before creating a new one."
                    }
        
        result = db_create_ticket(
            title=title,
            description=description,
            assignee=assignee or None,
            priority=priority,
            status=status
        )

        ticket_id = result.get('ticket_id')
        
        # Update embedding for new ticket if possible
        if ticket_id is not None:
            embedding = _embed_text(f"{title}\n{description}")
            if embedding:
                try:
                    update_ticket_embedding(ticket_id=str(ticket_id), embedding=embedding)
                except Exception as e:
                    print(f"⚠ Failed to store embedding for ticket {ticket_id}: {e}")

        return {"success": True, "ticket_id": ticket_id, "message": f"Ticket created successfully"}
    except Exception as e:
        return {"error": str(e), "ticket_id": None}


def get_current_datetime() -> dict:
    """Get current date and time"""
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timestamp": now.timestamp()
    }


def calculate(expression: str) -> dict:
    """Perform basic mathematical calculations"""
    try:
        # Safe evaluation of math expressions
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression", "result": None}
        
        result = eval(expression)
        return {"success": True, "expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "result": None}


# ============================================================================
# External Search / API Tools
# ============================================================================

_SEARCH_STOPWORDS = {
    "i", "me", "my", "mine", "we", "our", "you", "your", "the", "a", "an", "to", "for",
    "after", "before", "from", "this", "that", "it", "is", "are", "am", "was", "were", "be",
    "do", "does", "did", "how", "what", "why", "can", "could", "should", "would", "please",
    "needed", "need", "help", "fix", "issue", "problem", "getting", "not", "with", "and", "or",
}


def _tokenize_for_search(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9\+#._-]+", (text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in _SEARCH_STOPWORDS]


def _build_issue_search_query(query: str) -> str:
    """Build a focused search query from noisy user text."""
    q = (query or "").strip()
    q_lower = q.lower()
    focus_terms = []

    # Preserve key technical phrases
    phrase_patterns = [
        r"windows update", r"blue screen", r"slow pc", r"cpu usage", r"disk usage",
        r"memory leak", r"connection timeout", r"permission denied", r"module not found",
        r"vpn", r"wifi", r"network", r"docker", r"python", r"postgres", r"database",
    ]
    for pattern in phrase_patterns:
        if re.search(pattern, q_lower):
            focus_terms.append(pattern)

    # Add informative single terms
    tokens = _tokenize_for_search(q)
    for t in tokens:
        if t not in focus_terms:
            focus_terms.append(t)
        if len(focus_terms) >= 8:
            break

    if not focus_terms:
        return q
    return " ".join(focus_terms)


def _score_search_result(result: dict, query: str) -> int:
    """Simple lexical relevance score between query and search result."""
    query_tokens = set(_tokenize_for_search(query))
    if not query_tokens:
        return 1

    title = (result.get("title") or "").lower()
    snippet = (result.get("snippet") or "").lower()
    haystack = f"{title} {snippet}"

    score = 0
    for token in query_tokens:
        if token in title:
            score += 3
        elif token in haystack:
            score += 1

    # Strongly prefer platform/domain matching when user mentions it.
    if "windows" in query_tokens and "windows" in haystack:
        score += 2
    if "python" in query_tokens and "python" in haystack:
        score += 2
    if "vpn" in query_tokens and "vpn" in haystack:
        score += 2

    return score


def _filter_relevant_results(results: List[dict], query: str, max_items: int = 5) -> List[dict]:
    if not results:
        return []
    ranked = sorted(results, key=lambda r: _score_search_result(r, query), reverse=True)
    # Keep only meaningfully related results; fallback to top 2 if everything is weak.
    strong = [r for r in ranked if _score_search_result(r, query) >= 3]
    if strong:
        return strong[:max_items]
    return ranked[:min(2, len(ranked))]

def search_google(query: str, num_results: int = 5) -> dict:
    """Search the web using DuckDuckGo HTML search (no API key needed).
    Returns a list of results with title and link.
    """
    if requests is None:
        return {"error": "Python package 'requests' not available.", "results": []}

    try:
        # Use DuckDuckGo HTML search (more reliable than API)
        search_url = "https://html.duckduckgo.com/html/"
        focused_query = _build_issue_search_query(query)
        params = {"q": focused_query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.post(search_url, data=params, headers=headers, timeout=6)
        resp.raise_for_status()
        
        results = []
        html = resp.text
        
        # Parse results using regex (avoid BeautifulSoup dependency)
        # Look for result links and titles
        result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'
        
        links = re.findall(result_pattern, html)
        snippets = re.findall(snippet_pattern, html)
        
        for i, (link, title) in enumerate(links[:num_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            # Clean up HTML entities
            title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            snippet = snippet.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            results.append({
                "title": title.strip(),
                "link": link,
                "snippet": snippet.strip()
            })
        
        if not results:
            # Fallback to Instant Answer API
            api_params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            api_resp = requests.get("https://api.duckduckgo.com/", params=api_params, headers=headers, timeout=6)
            if api_resp.status_code == 200:
                data = api_resp.json()
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", "Result"),
                        "link": data.get("AbstractURL", ""),
                        "snippet": data.get("Abstract", "")
                    })
                for topic in data.get("RelatedTopics", [])[:num_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text", "")[:80],
                            "link": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })
        
        filtered = _filter_relevant_results(results, focused_query, max_items=num_results)
        if filtered:
            return {"success": True, "results": filtered}
        else:
            return {
                "success": True,
                "results": [],
                "message": f"No relevant web results found for '{query}'. Try refining the issue details."
            }
    except Exception as e:
        return {"error": str(e), "results": []}


def query_stackexchange(query: str, site: str = "stackoverflow", num: int = 5) -> dict:
    """Query StackExchange API (StackOverflow) for relevant Q&A.
    Returns list of question titles and links.
    """
    if requests is None:
        return {"error": "Python package 'requests' not available.", "results": []}

    try:
        focused_query = _build_issue_search_query(query)
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": focused_query,
            "site": site,
            "pagesize": max(10, num)
        }
        resp = requests.get("https://api.stackexchange.com/2.3/search/advanced", params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for it in data.get("items", []):
            results.append({"title": it.get("title"), "link": it.get("link"), "score": it.get("score")})
        filtered = _filter_relevant_results(results, focused_query, max_items=num)
        return {"success": True, "results": filtered}
    except Exception as e:
        return {"error": str(e), "results": []}


def github_mcp_search(query: str, repo: Optional[str] = None, num: int = 5) -> dict:
    """Search GitHub issues/pulls using the GitHub Search API (works without API key, but rate limited)."""
    if requests is None:
        return {"error": "Python package 'requests' not available.", "results": []}

    try:
        q = f"{query} in:title,body"
        if repo:
            q += f" repo:{repo}"
        params = {"q": q, "per_page": num}
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "SoftwareBugAssistant/1.0"}
        
        # Add token if available for higher rate limits
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        
        resp = requests.get("https://api.github.com/search/issues", params=params, headers=headers, timeout=6)
        
        # Handle rate limiting gracefully
        if resp.status_code == 403:
            return {
                "success": False,
                "error": "GitHub API rate limit reached. Results may be limited.",
                "results": [],
                "message": "Try again in a few minutes or set GITHUB_PERSONAL_ACCESS_TOKEN for higher limits."
            }
        
        resp.raise_for_status()
        data = resp.json()
        results = []
        for it in data.get("items", [])[:num]:
            results.append({"title": it.get("title"), "link": it.get("html_url"), "state": it.get("state")})
        return {"success": True, "results": results}
    except Exception as e:
        return {"error": str(e), "results": []}


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """Registry of all available tools for the local LLM"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools"""
        
        # Database Tools
        self.register(Tool(
            name="get_tickets_by_status",
            description="Retrieve tickets filtered by their status. Use this when the user asks for tickets with a specific status like 'open', 'closed', 'in progress', or 'resolved'.",
            parameters=[
                ToolParameter("status", "string", "The status to filter by (e.g., 'Open', 'In Progress', 'Closed', 'Resolved')", 
                             enum=["Open", "In Progress", "Closed", "Resolved"])
            ],
            function=get_tickets_by_status,
            category="database"
        ))
        
        self.register(Tool(
            name="get_tickets_by_priority",
            description="Retrieve tickets filtered by priority level. Use this when the user asks for critical, high, medium, or low priority tickets.",
            parameters=[
                ToolParameter("priority", "string", "The priority to filter by",
                             enum=["P0 - Critical", "P1 - High", "P2 - Medium", "P3 - Low"])
            ],
            function=get_tickets_by_priority,
            category="database"
        ))
        
        self.register(Tool(
            name="get_ticket_by_id",
            description="Retrieve a specific ticket by its unique ID. Use this when the user asks about a specific ticket number.",
            parameters=[
                ToolParameter("ticket_id", "string", "The unique ticket ID (e.g., 'TKT-001')")
            ],
            function=get_ticket_by_id,
            category="database"
        ))
        
        self.register(Tool(
            name="get_tickets_by_assignee",
            description="Retrieve tickets assigned to a specific person. Use this when the user asks for tickets assigned to someone.",
            parameters=[
                ToolParameter("assignee", "string", "The email or name of the assignee")
            ],
            function=get_tickets_by_assignee,
            category="database"
        ))
        
        self.register(Tool(
            name="search_tickets",
            description="Search for tickets using semantic similarity. Use this when the user wants to find tickets related to a topic or similar to a description.",
            parameters=[
                ToolParameter("query", "string", "The search query to find similar tickets")
            ],
            function=search_tickets,
            category="database"
        ))
        
        self.register(Tool(
            name="create_ticket",
            description="Create a new bug ticket. Use this when the user wants to create, add, or submit a new ticket.",
            parameters=[
                ToolParameter("title", "string", "The title of the ticket"),
                ToolParameter("description", "string", "Detailed description of the bug or issue"),
                ToolParameter("assignee", "string", "Email of the person to assign (optional)", required=False, default=""),
                ToolParameter("priority", "string", "Priority level", required=False, default="P3 - Low",
                             enum=["P0 - Critical", "P1 - High", "P2 - Medium", "P3 - Low"]),
                ToolParameter("status", "string", "Initial status", required=False, default="Open",
                             enum=["Open", "In Progress", "Closed", "Resolved"])
            ],
            function=create_ticket,
            category="database"
        ))
        
        self.register(Tool(
            name="update_ticket_status",
            description="Update the status of an existing ticket. Use this when the user wants to change a ticket's status.",
            parameters=[
                ToolParameter("ticket_id", "string", "The ticket ID to update"),
                ToolParameter("status", "string", "The new status",
                             enum=["Open", "In Progress", "Closed", "Resolved"])
            ],
            function=update_ticket_status,
            category="database"
        ))
        
        self.register(Tool(
            name="update_ticket_priority",
            description="Update the priority of an existing ticket. Use this when the user wants to change a ticket's priority.",
            parameters=[
                ToolParameter("ticket_id", "string", "The ticket ID to update"),
                ToolParameter("priority", "string", "The new priority",
                             enum=["P0 - Critical", "P1 - High", "P2 - Medium", "P3 - Low"])
            ],
            function=update_ticket_priority,
            category="database"
        ))
        
        self.register(Tool(
            name="get_tickets_by_date_range",
            description="Retrieve tickets created or updated within a date range. Use this when the user asks for tickets from a specific time period.",
            parameters=[
                ToolParameter("start_date", "string", "Start date in YYYY-MM-DD format"),
                ToolParameter("end_date", "string", "End date in YYYY-MM-DD format"),
                ToolParameter("date_field", "string", "Which date to filter by", required=False, default="creation_time",
                             enum=["creation_time", "updated_time"])
            ],
            function=get_tickets_by_date_range,
            category="database"
        ))
        
        # Utility Tools
        self.register(Tool(
            name="get_current_datetime",
            description="Get the current date and time. Use this when the user asks about the current date, time, or day.",
            parameters=[],
            function=get_current_datetime,
            category="utility"
        ))
        
        self.register(Tool(
            name="calculate",
            description="Perform mathematical calculations. Use this for any math operations.",
            parameters=[
                ToolParameter("expression", "string", "The mathematical expression to evaluate (e.g., '2 + 2', '100 * 5')")
            ],
            function=calculate,
            category="utility"
        ))

        # External Search / Knowledge Tools
        self.register(Tool(
            name="google_search",
            description="Search the web using SerpAPI/Google Search. Returns top results with title, link and snippet.",
            parameters=[
                ToolParameter("query", "string", "Search query string"),
                ToolParameter("num_results", "integer", "Number of results to return", required=False, default=5)
            ],
            function=search_google,
            category="external"
        ))

        self.register(Tool(
            name="stackexchange_search",
            description="Search StackExchange (e.g., StackOverflow) for relevant Q&A.",
            parameters=[
                ToolParameter("query", "string", "Search query string"),
                ToolParameter("site", "string", "StackExchange site to query (default: 'stackoverflow')", required=False, default="stackoverflow")
            ],
            function=query_stackexchange,
            category="external"
        ))

        self.register(Tool(
            name="github_mcp_search",
            description="Search GitHub issues/pull requests using GitHub Search API. Requires GITHUB_PERSONAL_ACCESS_TOKEN env var.",
            parameters=[
                ToolParameter("query", "string", "Search query string"),
                ToolParameter("repo", "string", "Optional repo owner/name to restrict search (e.g., 'owner/repo')", required=False, default=""),
                ToolParameter("num", "integer", "Number of results to return", required=False, default=5)
            ],
            function=github_mcp_search,
            category="external"
        ))
    
    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names"""
        return list(self.tools.keys())
    
    def get_all_schemas(self) -> List[dict]:
        """Get schemas for all tools"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        """Get a formatted description of all tools for the LLM prompt"""
        descriptions = []
        for name, tool in self.tools.items():
            params_str = ", ".join([
                f"{p.name}: {p.type}" + (f" (optional, default={p.default})" if not p.required else "")
                for p in tool.parameters
            ])
            descriptions.append(f"- **{name}**({params_str}): {tool.description}")
        return "\n".join(descriptions)
    
    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call"""
        tool = self.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_call.tool_name}"
            )
        
        try:
            result = tool.function(**tool_call.arguments)
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=str(e)
            )


# ============================================================================
# Tool Call Parser
# ============================================================================

class ToolCallParser:
    """Parse tool calls from LLM output"""
    
    # Pattern to match tool calls in various formats
    TOOL_CALL_PATTERNS = [
        # JSON format: {"tool": "name", "arguments": {...}}
        r'\{["\']?tool["\']?\s*:\s*["\'](\w+)["\'].*?["\']?arguments["\']?\s*:\s*(\{[^}]+\})\s*\}',
        # Function call format: tool_name(arg1="value1", arg2="value2")
        r'(\w+)\(([^)]*)\)',
        # TOOL_CALL format: TOOL_CALL: tool_name {"arg": "value"}
        r'TOOL_CALL:\s*(\w+)\s*(\{[^}]+\})',
        # Action format: Action: tool_name\nArguments: {...}
        r'Action:\s*(\w+).*?Arguments?:\s*(\{[^}]+\})',
    ]
    
    @classmethod
    def parse(cls, text: str, available_tools: List[str]) -> List[ToolCall]:
        """Parse tool calls from LLM output"""
        tool_calls = []
        
        # Try JSON format first
        json_calls = cls._parse_json_format(text, available_tools)
        if json_calls:
            return json_calls
        
        # Try function call format
        func_calls = cls._parse_function_format(text, available_tools)
        if func_calls:
            return func_calls
        
        # Try TOOL_CALL format
        tc_calls = cls._parse_tool_call_format(text, available_tools)
        if tc_calls:
            return tc_calls
        
        return tool_calls
    
    @classmethod
    def _parse_json_format(cls, text: str, available_tools: List[str]) -> List[ToolCall]:
        """Parse JSON formatted tool calls"""
        tool_calls = []
        
        # Look for JSON objects with tool and arguments
        try:
            # Try to find JSON blocks
            json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    data = json.loads(match)
                    if "tool" in data and data["tool"] in available_tools:
                        tool_calls.append(ToolCall(
                            tool_name=data["tool"],
                            arguments=data.get("arguments", {})
                        ))
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return tool_calls
    
    @classmethod
    def _parse_function_format(cls, text: str, available_tools: List[str]) -> List[ToolCall]:
        """Parse function-style tool calls: tool_name(arg1="value1")"""
        tool_calls = []
        
        for tool_name in available_tools:
            # Pattern: tool_name(...)
            pattern = rf'{tool_name}\s*\(([^)]*)\)'
            matches = re.findall(pattern, text)
            
            for args_str in matches:
                arguments = cls._parse_arguments(args_str)
                tool_calls.append(ToolCall(tool_name=tool_name, arguments=arguments))
        
        return tool_calls
    
    @classmethod
    def _parse_tool_call_format(cls, text: str, available_tools: List[str]) -> List[ToolCall]:
        """Parse TOOL_CALL: format and tool_name {args} format"""
        tool_calls = []
        
        # Pattern 1: TOOL_CALL: tool_name {args}
        pattern = r'TOOL_CALL:\s*(\w+)\s*(\{[^}]+\})?'
        matches = re.findall(pattern, text)
        
        for tool_name, args_json in matches:
            if tool_name in available_tools:
                arguments = {}
                if args_json:
                    try:
                        arguments = json.loads(args_json)
                    except json.JSONDecodeError:
                        pass
                tool_calls.append(ToolCall(tool_name=tool_name, arguments=arguments))
        
        # Pattern 2: tool_name {"arg": "value"} (without TOOL_CALL: prefix)
        for tool_name in available_tools:
            # Match tool_name followed by JSON object
            pattern2 = rf'\b{tool_name}\s*(\{{[^{{}}]+\}})'
            matches2 = re.findall(pattern2, text)
            for args_json in matches2:
                try:
                    arguments = json.loads(args_json)
                    # Avoid duplicates
                    if not any(tc.tool_name == tool_name and tc.arguments == arguments for tc in tool_calls):
                        tool_calls.append(ToolCall(tool_name=tool_name, arguments=arguments))
                except json.JSONDecodeError:
                    pass
        
        return tool_calls
    
    @classmethod
    def _parse_arguments(cls, args_str: str) -> dict:
        """Parse argument string into dict"""
        arguments = {}
        
        if not args_str.strip():
            return arguments
        
        # Try JSON first
        try:
            return json.loads("{" + args_str + "}")
        except json.JSONDecodeError:
            pass
        
        # Parse key=value pairs
        # Pattern: key="value" or key='value' or key=value
        pattern = r'(\w+)\s*=\s*["\']?([^"\'=,]+)["\']?'
        matches = re.findall(pattern, args_str)
        
        for key, value in matches:
            # Try to convert to appropriate type
            value = value.strip()
            if value.lower() == "true":
                arguments[key] = True
            elif value.lower() == "false":
                arguments[key] = False
            elif value.isdigit():
                arguments[key] = int(value)
            else:
                arguments[key] = value
        
        return arguments


# ============================================================================
# Local LLM with Tool Calling
# ============================================================================

class LocalLLMWithTools:
    """
    Local Gemma-2B with function calling capabilities.
    Implements a ReAct-style reasoning loop.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        from pathlib import Path
        MODEL_DIR = Path(__file__).parent.parent.parent / "models"
        GEMMA_MODEL = MODEL_DIR / "gemma-2-2b-it-Q4_K_M.gguf"
        
        self.model = None
        self.model_path = model_path or str(GEMMA_MODEL)
        self.tool_registry = ToolRegistry()
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma model"""
        if not os.path.exists(self.model_path):
            print(f"⚠ Model not found: {self.model_path}")
            return
        
        try:
            from llama_cpp import Llama
            
            print("Loading Gemma-2B with tool calling support...")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=4,
                n_gpu_layers=0,
                verbose=False
            )
            print("✓ Gemma-2B with tools ready!")
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions"""
        tools_desc = self.tool_registry.get_tools_description()
        
        return f"""You are a skilled software bug triaging expert at Tandion IT Software company. You have access to the following tools to help users:

**Available Tools:**
{tools_desc}

**Instructions:**
1. Read the user's message carefully and decide the SINGLE best tool to call.
2. Always output a tool call using EXACTLY this format (one per line, no extra text before it):
   TOOL_CALL: tool_name {{"param1": "value1", "param2": "value2"}}
3. Do NOT explain what you are going to do. Just output the TOOL_CALL line immediately.

**Intent → Tool mapping:**
- User wants to CLOSE / RESOLVE / UPDATE status of a ticket → update_ticket_status
- User wants to CHANGE PRIORITY of a ticket → update_ticket_priority
- User wants to SEE / LIST / SHOW tickets → get_tickets_by_status or get_tickets_by_priority
- User wants to FIND a specific ticket by ID → get_ticket_by_id
- User wants to SEARCH tickets by keyword/text → search_tickets
- User wants to CREATE / REPORT a new ticket → create_ticket
- User has a TECHNICAL PROBLEM (error, crash, not working) → google_search AND stackexchange_search

**Examples:**
- User: "close ticket 27"
  TOOL_CALL: update_ticket_status {{"ticket_id": "27", "status": "Closed"}}

- User: "I needed to close the ticket for my cpu is getting slow"
  TOOL_CALL: update_ticket_status {{"ticket_id": "22", "status": "Closed"}}

- User: "mark ticket #5 as resolved"
  TOOL_CALL: update_ticket_status {{"ticket_id": "5", "status": "Resolved"}}

- User: "resolve ticket 10"
  TOOL_CALL: update_ticket_status {{"ticket_id": "10", "status": "Resolved"}}

- User: "set ticket 3 to in progress"
  TOOL_CALL: update_ticket_status {{"ticket_id": "3", "status": "In Progress"}}

- User: "change priority of ticket 7 to critical"
  TOOL_CALL: update_ticket_priority {{"ticket_id": "7", "priority": "P0 - Critical"}}

- User: "show all open tickets"
  TOOL_CALL: get_tickets_by_status {{"status": "Open"}}

- User: "list critical bugs"
  TOOL_CALL: get_tickets_by_priority {{"priority": "P0 - Critical"}}

- User: "show ticket #12"
  TOOL_CALL: get_ticket_by_id {{"ticket_id": "12"}}

- User: "search for login issues"
  TOOL_CALL: search_tickets {{"query": "login issues"}}

- User: "create a ticket for database crash"
  TOOL_CALL: create_ticket {{"title": "Database crash", "description": "Database crash reported by user", "priority": "P1 - High", "status": "Open"}}

- User: "my VPN is not working"
  TOOL_CALL: google_search {{"query": "VPN not connecting troubleshooting solutions"}}

- User: "getting python import error"
  TOOL_CALL: stackexchange_search {{"query": "python import error module not found"}}
"""

    def _extract_json_block(self, text: str) -> Optional[dict]:
        """Extract a JSON object from a model response."""
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None
        block = match.group(0)
        try:
            return json.loads(block)
        except Exception:
            return None

    def _normalize_priority(self, value: str) -> str:
        """Normalize priority labels to DB format."""
        if not value:
            return ""
        v = value.lower().strip()
        if v in ["p0", "critical"]:
            return "P0 - Critical"
        if v in ["p1", "high"]:
            return "P1 - High"
        if v in ["p2", "medium"]:
            return "P2 - Medium"
        if v in ["p3", "low"]:
            return "P3 - Low"
        return value

    def _normalize_status(self, value: str) -> str:
        """Normalize status labels to DB format."""
        if not value:
            return ""
        v = value.lower().strip()
        if v in ["open"]:
            return "Open"
        if v in ["closed", "close"]:
            return "Closed"
        if v in ["resolved", "resolve", "done", "fixed"]:
            return "Resolved"
        if v in ["in progress", "in-progress", "progress"]:
            return "In Progress"
        return value

    def _infer_intent(self, user_message: str) -> dict:
        """Infer user intent and required slots using the local model."""
        prompt = f"""You are an intent classifier for a bug triage assistant.
Return a single JSON object with:
  intent: one of [create_ticket, update_status, update_priority, view_tickets, troubleshoot, unknown]
  confidence: 0-1
  slots: ticket_id, status, priority, assignee, query, issue

User message: "{user_message}"

Rules:
- If user asks to fix or solve a problem, use troubleshoot.
- If user asks to create/open/report a ticket, use create_ticket.
- If user asks to close/resolve/update status of a ticket, use update_status.
- If user asks to change priority, use update_priority.
- If user asks to list/show/search tickets, use view_tickets.
- If unsure, use unknown.

Only output JSON."""

        try:
            response = self._generate(prompt, max_tokens=256)
            data = self._extract_json_block(response) or {}
        except Exception:
            data = {}

        intent = str(data.get("intent", "unknown"))
        confidence = float(data.get("confidence", 0)) if str(data.get("confidence", "")).replace('.', '', 1).isdigit() else 0
        slots = data.get("slots", {}) if isinstance(data.get("slots", {}), dict) else {}

        allowed_intents = {
            "create_ticket",
            "update_status",
            "update_priority",
            "view_tickets",
            "troubleshoot",
            "unknown",
        }
        if intent not in allowed_intents:
            intent = "unknown"

        # Always extract common slots from text
        q = user_message.lower()
        ticket_id_match = re.search(r'(?:ticket\s*#?|#)\s*(\d+)', q)
        if ticket_id_match:
            slots["ticket_id"] = ticket_id_match.group(1)

        priority_match = re.search(r'\b(p0|p1|p2|p3|critical|high|medium|low)\b', q)
        if priority_match:
            slots["priority"] = self._normalize_priority(priority_match.group(1))

        status_match = re.search(r'\b(open|closed|close|resolved|resolve|in progress|in-progress|done|fixed)\b', q)
        if status_match:
            slots["status"] = self._normalize_status(status_match.group(1))

        # Heuristic fallback if model output is weak
        if intent == "unknown" or confidence < 0.4:
            if re.search(r'\b(create|open|report|log|file|raise)\b', q):
                intent = "create_ticket"
            elif re.search(r'\b(priority|critical|p0|p1|p2|p3|high|medium|low)\b', q) and re.search(r'\bticket\b', q):
                intent = "update_priority"
            elif re.search(r'\b(close|closed|resolve|resolved|in progress|update status)\b', q) and re.search(r'\bticket\b', q):
                intent = "update_status"
            elif self._is_ticket_viewing_request(user_message):
                intent = "view_tickets"
            elif self._is_technical_issue_query(user_message):
                intent = "troubleshoot"

        # Normalize slots
        if slots.get("priority"):
            slots["priority"] = self._normalize_priority(str(slots.get("priority")))
        if slots.get("status"):
            slots["status"] = self._normalize_status(str(slots.get("status")))

        # If user mentions a ticket id + priority/status, prefer update intents
        if slots.get("ticket_id") and slots.get("priority"):
            intent = "update_priority"
        elif slots.get("ticket_id") and slots.get("status"):
            intent = "update_status"
        elif slots.get("priority") and re.search(r'\b(show|list|get|display|view)\b', q):
            intent = "view_tickets"
        elif re.search(r'\b(show|list|get|display|view)\b', q) and re.search(r'\b(ticket|tickets|bugs|issues)\b', q):
            intent = "view_tickets"

        return {
            "intent": intent,
            "confidence": confidence,
            "slots": slots,
        }
    
    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from the model"""
        if not self.is_available():
            raise RuntimeError("Model not loaded")
        
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        output = self.model(
            formatted,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        
        return output["choices"][0]["text"].strip()
    
    def _format_tool_results(self, results: List[ToolResult]) -> str:
        """Format tool results for the model"""
        formatted = []
        for result in results:
            if result.success:
                # Format nicely based on result type
                data = result.result
                if isinstance(data, dict):
                    if "tickets" in data:
                        tickets = data.get("tickets", [])
                        count = data.get("count", len(tickets) if isinstance(tickets, list) else 0)
                        if isinstance(tickets, list) and len(tickets) > 0:
                            formatted.append(f"**{result.tool_name} returned {len(tickets)} ticket(s):**")
                            for i, ticket in enumerate(tickets):  # Show all tickets
                                if isinstance(ticket, dict):
                                    tid = ticket.get("ticket_id", "N/A")
                                    title = ticket.get("title", "N/A")
                                    status = ticket.get("status", "N/A")
                                    priority = ticket.get("priority", "N/A")
                                    assignee = ticket.get("assignee", "Unassigned")
                                    if "@" in str(assignee):
                                        assignee = assignee.split("@")[0]
                                    formatted.append(f"  {i+1}. **[#{tid}]** {title}")
                                    formatted.append(f"      Status: {status} | Priority: {priority} | Assignee: {assignee}")
                        else:
                            formatted.append(f"**{result.tool_name}:** No tickets found matching the criteria.")
                    elif "ticket" in data and data.get("ticket"):
                        ticket = data["ticket"]
                        if isinstance(ticket, dict):
                            formatted.append(f"**Ticket Details:**")
                            for key, val in ticket.items():
                                formatted.append(f"  - {key}: {val}")
                        else:
                            formatted.append(f"**{result.tool_name}:** {ticket}")
                    elif "message" in data:
                        formatted.append(f"**{result.tool_name}:** {data['message']}")
                    elif "error" in data:
                        formatted.append(f"**{result.tool_name} Error:** {data['error']}")
                    else:
                        formatted.append(f"**{result.tool_name}:** {json.dumps(data, indent=2)}")
                else:
                    formatted.append(f"**{result.tool_name}:** {data}")
            else:
                formatted.append(f"**{result.tool_name} failed:** {result.error}")
        
        return "\n".join(formatted)
    
    def _get_ai_troubleshooting_response(self, user_query: str) -> str:
        """Generate intelligent AI troubleshooting response by analyzing the specific error"""
        query_lower = user_query.lower()
        
        # =====================================================================
        # INTELLIGENT ERROR ANALYSIS - Parse and understand specific errors
        # =====================================================================
        
        # Git-specific errors
        if 'git' in query_lower or 'remote' in query_lower or 'origin' in query_lower or 'fatal:' in query_lower:
            return self._analyze_git_error(user_query)
        
        # Docker errors
        if 'docker' in query_lower or 'container' in query_lower or 'image' in query_lower:
            return self._analyze_docker_error(user_query)
        
        # npm/Node.js errors
        if 'npm' in query_lower or 'node' in query_lower or 'yarn' in query_lower or 'enoent' in query_lower:
            return self._analyze_npm_error(user_query)
        
        # Python errors
        if 'python' in query_lower or 'pip' in query_lower or 'modulenotfound' in query_lower or 'importerror' in query_lower or 'traceback' in query_lower:
            return self._analyze_python_error(user_query)
        
        # Database errors
        if 'sql' in query_lower or 'database' in query_lower or 'mysql' in query_lower or 'postgres' in query_lower or 'mongodb' in query_lower:
            return self._analyze_database_error(user_query)
        
        # Network/Connection errors
        if 'connection refused' in query_lower or 'timeout' in query_lower or 'econnrefused' in query_lower or 'network' in query_lower:
            return self._analyze_network_error(user_query)
        
        # Permission errors
        if 'permission' in query_lower or 'access denied' in query_lower or 'eacces' in query_lower or 'forbidden' in query_lower:
            return self._analyze_permission_error(user_query)
        
        # VPN issues
        if 'vpn' in query_lower:
            return self._analyze_vpn_error(user_query)
        
        # Authentication errors
        if 'password' in query_lower or 'login' in query_lower or 'authentication' in query_lower or '401' in query_lower or '403' in query_lower:
            return self._analyze_auth_error(user_query)
        
        # SSL/TLS errors
        if 'ssl' in query_lower or 'certificate' in query_lower or 'https' in query_lower or 'tls' in query_lower:
            return self._analyze_ssl_error(user_query)
        
        # Windows-specific errors (0x codes)
        if '0x' in query_lower or 'windows error' in query_lower:
            return self._analyze_windows_error(user_query)
        
        # API errors
        if 'api' in query_lower or 'http' in query_lower or '500' in query_lower or '404' in query_lower:
            return self._analyze_api_error(user_query)
        
        # Hardware / RAM / Memory issues
        if any(kw in query_lower for kw in ['ram', 'memory', 'bsod', 'blue screen', 'memory management', 'crash']):
            return self._analyze_hardware_error(user_query)
        
        # If no specific pattern matched, do intelligent general analysis
        return self._analyze_generic_error(user_query)

    def _analyze_git_error(self, query: str) -> str:
        """Intelligent analysis of Git errors"""
        query_lower = query.lower()
        
        # fatal: remote origin already exists
        if 'remote' in query_lower and ('already exists' in query_lower or 'origin' in query_lower):
            return """## 🔍 Error Analysis: `fatal: remote origin already exists`

### 🎯 What This Error Means
This error occurs because you're trying to add a remote named "origin" to your Git repository, but a remote with that name **already exists**. Git only allows one remote per name.

### 🔧 Root Cause
- You previously ran `git remote add origin <url>` 
- The remote "origin" is already configured in `.git/config`
- You may be trying to connect to a different repository URL

### ✅ Solutions (Choose One)

**Option 1: Update the existing remote URL** (Most Common)
```bash
git remote set-url origin <new-repository-url>
```

**Option 2: Remove and re-add the remote**
```bash
git remote remove origin
git remote add origin <repository-url>
```

**Option 3: Use a different remote name**
```bash
git remote add upstream <repository-url>
```

**Option 4: View current remote configuration**
```bash
git remote -v
```

### 💡 Prevention Tips
- Always check existing remotes with `git remote -v` before adding
- Use `set-url` to change URLs instead of add/remove

---
"""
        
        # merge conflicts
        if 'merge' in query_lower and 'conflict' in query_lower:
            return """## 🔍 Error Analysis: Git Merge Conflicts

### 🎯 What This Error Means
A merge conflict happens when Git cannot automatically combine changes from different branches because the **same lines** were modified differently in both branches.

### 🔧 Root Cause
- You and another developer (or another branch) modified the same file
- Git doesn't know which version to keep
- This is normal in collaborative development!

### ✅ Step-by-Step Solution

**Step 1: Identify conflicted files**
```bash
git status
```
Look for files marked as "both modified"

**Step 2: Open the conflicted file**
Look for conflict markers:
```
<<<<<<< HEAD
your changes
=======
their changes
>>>>>>> branch-name
```

**Step 3: Resolve the conflict**
- Decide which changes to keep (yours, theirs, or a combination)
- Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
- Save the file

**Step 4: Mark as resolved and commit**
```bash
git add <resolved-file>
git commit -m "Resolved merge conflict in <file>"
```

### 💡 Pro Tips
- Use `git mergetool` for a visual conflict resolver
- Communicate with your team to avoid parallel edits

---
"""
        
        # push rejected
        if 'push' in query_lower and ('rejected' in query_lower or 'non-fast-forward' in query_lower):
            return """## 🔍 Error Analysis: Git Push Rejected (Non-Fast-Forward)

### 🎯 What This Error Means
Your push was rejected because the remote branch has commits that your local branch doesn't have. Git prevents you from overwriting those commits.

### 🔧 Root Cause
- Someone else pushed changes to the same branch
- You need to integrate their changes before pushing yours

### ✅ Solutions

**Option 1: Pull and merge (Recommended)**
```bash
git pull origin <branch-name>
# Resolve any conflicts if needed
git push origin <branch-name>
```

**Option 2: Pull with rebase (Cleaner history)**
```bash
git pull --rebase origin <branch-name>
git push origin <branch-name>
```

**Option 3: Force push (⚠️ DANGEROUS - Only if you're sure)**
```bash
git push --force origin <branch-name>
```
⚠️ **Warning**: Force push overwrites remote history. Only use on personal branches!

### 💡 Best Practice
Always `git pull` before starting new work to minimize conflicts.

---
"""

        # Default git error
        return """## 🔍 Git Error Analysis

### 🎯 Analyzing Your Git Issue

Based on your query, here are intelligent troubleshooting steps:

### 1. 🔍 Check Git Status
```bash
git status
```
This shows the current state of your repository.

### 2. 📋 View Git Log
```bash
git log --oneline -10
```
See recent commits to understand the history.

### 3. 🌐 Check Remote Configuration
```bash
git remote -v
```
Verify your remote repository settings.

### 4. 🔄 Common Fixes
```bash
# Reset to last commit (⚠️ loses uncommitted changes)
git reset --hard HEAD

# Stash changes temporarily
git stash
git pull
git stash pop
```

### 💡 Need More Help?
Please share the **exact error message** for a more specific solution.

---
"""

    def _analyze_docker_error(self, query: str) -> str:
        """Intelligent analysis of Docker errors"""
        query_lower = query.lower()
        
        if 'port' in query_lower and ('already' in query_lower or 'in use' in query_lower or 'bind' in query_lower):
            return """## 🔍 Error Analysis: Docker Port Already in Use

### 🎯 What This Error Means
Another process (container or application) is already using the port you're trying to bind.

### 🔧 Root Cause
- A previous container is still running on that port
- Another application (like a local server) is using the port

### ✅ Solutions

**Step 1: Find what's using the port**
```bash
# Linux/Mac
lsof -i :<port-number>
netstat -tulpn | grep <port-number>

# Windows
netstat -ano | findstr :<port-number>
```

**Step 2: Stop the conflicting container**
```bash
docker ps  # List running containers
docker stop <container-id>
```

**Step 3: Or use a different port**
```bash
docker run -p 8081:80 <image>  # Map to different host port
```

---
"""
        
        if 'no such image' in query_lower or 'not found' in query_lower:
            return """## 🔍 Error Analysis: Docker Image Not Found

### 🎯 What This Error Means
Docker cannot find the image you specified, either locally or on Docker Hub.

### 🔧 Root Cause
- Image name is misspelled
- Image doesn't exist in the registry
- You need to build the image first

### ✅ Solutions

**Step 1: Check image name spelling**
```bash
docker search <image-name>
```

**Step 2: Pull the image explicitly**
```bash
docker pull <image-name>:<tag>
```

**Step 3: If it's a local image, build it**
```bash
docker build -t <image-name> .
```

**Step 4: List local images**
```bash
docker images
```

---
"""

        return """## 🔍 Docker Error Analysis

### 🎯 Common Docker Troubleshooting

**Check Docker status:**
```bash
docker info
docker ps -a  # All containers
docker images # All images
```

**View container logs:**
```bash
docker logs <container-id>
```

**Clean up unused resources:**
```bash
docker system prune
```

---
"""

    def _analyze_npm_error(self, query: str) -> str:
        """Intelligent analysis of npm/Node.js errors"""
        query_lower = query.lower()
        
        if 'enoent' in query_lower or 'no such file' in query_lower:
            return """## 🔍 Error Analysis: ENOENT (No Such File or Directory)

### 🎯 What This Error Means
Node.js/npm cannot find a required file or directory.

### 🔧 Root Cause
- `package.json` is missing
- A required file path is incorrect
- `node_modules` wasn't installed

### ✅ Solutions

**Step 1: Ensure you're in the right directory**
```bash
ls  # Should see package.json
```

**Step 2: Install dependencies**
```bash
npm install
```

**Step 3: If package.json is missing**
```bash
npm init -y
```

---
"""
        
        if 'permission' in query_lower or 'eacces' in query_lower:
            return """## 🔍 Error Analysis: npm Permission Error (EACCES)

### 🎯 What This Error Means
npm doesn't have permission to write to the required directory.

### ✅ Solutions

**Option 1: Fix npm permissions (Recommended)**
```bash
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
# Add to PATH: export PATH=~/.npm-global/bin:$PATH
```

**Option 2: Use a Node version manager**
```bash
# Install nvm and use it to manage Node
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
```

**Option 3 (Not recommended): Use sudo**
```bash
sudo npm install -g <package>  # Avoid if possible
```

---
"""

        return """## 🔍 npm/Node.js Error Analysis

### 🎯 Common Solutions

**Clear npm cache:**
```bash
npm cache clean --force
```

**Reinstall dependencies:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Check Node.js version:**
```bash
node --version
npm --version
```

---
"""

    def _analyze_python_error(self, query: str) -> str:
        """Intelligent analysis of Python errors"""
        query_lower = query.lower()
        
        if 'modulenotfound' in query_lower or 'no module named' in query_lower:
            # Extract module name if possible
            import re
            module_match = re.search(r"no module named ['\"]?(\w+)", query_lower)
            module_name = module_match.group(1) if module_match else "<module-name>"
            
            return f"""## 🔍 Error Analysis: ModuleNotFoundError

### 🎯 What This Error Means
Python cannot find the module `{module_name}`. This usually means the package isn't installed or you're using the wrong Python environment.

### 🔧 Root Cause
- Package not installed in current environment
- Wrong Python interpreter selected
- Virtual environment not activated

### ✅ Solutions

**Step 1: Install the missing package**
```bash
pip install {module_name}
```

**Step 2: Check if you're in the right environment**
```bash
which python  # Linux/Mac
where python  # Windows
pip list | grep {module_name}
```

**Step 3: Activate virtual environment (if using one)**
```bash
# Linux/Mac
source venv/bin/activate

# Windows
.\\venv\\Scripts\\activate
```

**Step 4: If using conda**
```bash
conda activate <env-name>
conda install {module_name}
```

---
"""
        
        if 'syntaxerror' in query_lower:
            return """## 🔍 Error Analysis: SyntaxError

### 🎯 What This Error Means
Python found invalid syntax in your code that doesn't follow Python grammar rules.

### 🔧 Common Causes
- Missing colon `:` after if/for/def/class
- Mismatched parentheses, brackets, or quotes
- Using Python 3 syntax in Python 2 (or vice versa)
- Incorrect indentation

### ✅ How to Fix

**Check the line mentioned in the error**, common issues:
```python
# Missing colon
if x == 5:  # ✅ Correct
if x == 5   # ❌ Missing colon

# Mismatched quotes
print("Hello")  # ✅ Correct
print("Hello')  # ❌ Mismatched quotes

# Parentheses
print(sum([1,2,3]))  # ✅ Correct
print(sum([1,2,3])   # ❌ Missing closing )
```

---
"""

        if 'indentationerror' in query_lower:
            return """## 🔍 Error Analysis: IndentationError

### 🎯 What This Error Means
Python uses indentation to define code blocks. This error means your indentation is inconsistent.

### 🔧 Root Cause
- Mixed tabs and spaces
- Inconsistent indentation levels
- Copy-pasted code with different indentation

### ✅ Solutions

**Step 1: Use consistent indentation**
- Choose either tabs OR spaces (spaces are recommended)
- Use 4 spaces per indentation level

**Step 2: Configure your editor**
- Set editor to insert spaces when Tab is pressed
- Enable "Show whitespace" to see tabs vs spaces

**Step 3: Fix the file**
```bash
# Use autopep8 to fix indentation
pip install autopep8
autopep8 --in-place --aggressive <file.py>
```

---
"""

        return """## 🔍 Python Error Analysis

### 🎯 General Python Troubleshooting

**Check Python environment:**
```bash
python --version
pip --version
pip list
```

**Verify virtual environment:**
```bash
# Check which Python is active
which python  # Linux/Mac
where python  # Windows
```

**Install requirements:**
```bash
pip install -r requirements.txt
```

---
"""

    def _analyze_database_error(self, query: str) -> str:
        """Intelligent analysis of database errors"""
        query_lower = query.lower()
        
        if 'connection refused' in query_lower:
            return """## 🔍 Error Analysis: Database Connection Refused

### 🎯 What This Error Means
The database server is not accepting connections on the specified host/port.

### 🔧 Root Cause
- Database service not running
- Wrong host/port configuration
- Firewall blocking the connection

### ✅ Solutions

**Step 1: Check if database is running**
```bash
# PostgreSQL
sudo systemctl status postgresql

# MySQL
sudo systemctl status mysql

# MongoDB
sudo systemctl status mongod
```

**Step 2: Start the database service**
```bash
# PostgreSQL
sudo systemctl start postgresql

# MySQL
sudo systemctl start mysql
```

**Step 3: Verify connection settings**
- Host: Usually `localhost` or `127.0.0.1`
- Port: PostgreSQL=5432, MySQL=3306, MongoDB=27017

---
"""

        if 'access denied' in query_lower:
            return """## 🔍 Error Analysis: Database Access Denied

### 🎯 What This Error Means
Authentication failed - wrong username, password, or insufficient privileges.

### ✅ Solutions

**Step 1: Verify credentials**
- Check username and password
- Ensure user exists in database

**Step 2: Grant permissions (run as admin/root)**
```sql
-- MySQL
GRANT ALL PRIVILEGES ON database.* TO 'user'@'localhost';
FLUSH PRIVILEGES;

-- PostgreSQL
GRANT ALL PRIVILEGES ON DATABASE dbname TO username;
```

---
"""

        return """## 🔍 Database Error Analysis

### 🎯 Common Database Troubleshooting

**Check database service:**
```bash
sudo systemctl status <postgresql|mysql|mongod>
```

**Test connection:**
```bash
# PostgreSQL
psql -h localhost -U username -d database

# MySQL
mysql -h localhost -u username -p
```

---
"""

    def _analyze_network_error(self, query: str) -> str:
        """Intelligent analysis of network errors"""
        query_lower = query.lower()
        
        if 'connection refused' in query_lower or 'econnrefused' in query_lower:
            return """## 🔍 Error Analysis: Connection Refused

### 🎯 What This Error Means
The target server actively rejected the connection - no service is listening on that port.

### 🔧 Root Cause
- Service/server not running
- Wrong port number
- Firewall blocking the connection
- Server only listening on localhost

### ✅ Solutions

**Step 1: Check if service is running**
```bash
# Check listening ports
netstat -tulpn | grep <port>  # Linux
netstat -ano | findstr <port>  # Windows
```

**Step 2: Start the service**
- Start the application/server that should be listening

**Step 3: Check firewall**
```bash
# Linux
sudo ufw status
sudo ufw allow <port>

# Windows
netsh advfirewall firewall show rule name=all
```

---
"""

        if 'timeout' in query_lower:
            return """## 🔍 Error Analysis: Connection Timeout

### 🎯 What This Error Means
The connection attempt took too long and was abandoned.

### 🔧 Root Cause
- Network latency or packet loss
- Server overloaded/slow to respond
- Firewall silently dropping packets
- Wrong IP/hostname

### ✅ Solutions

**Step 1: Test basic connectivity**
```bash
ping <host>
traceroute <host>  # Linux/Mac
tracert <host>     # Windows
```

**Step 2: Check DNS resolution**
```bash
nslookup <hostname>
```

**Step 3: Increase timeout in your application**
- Configure longer timeout values in your code/config

---
"""

        return """## 🔍 Network Error Analysis

### 🎯 General Network Troubleshooting

**Test connectivity:**
```bash
ping <host>
curl -v <url>
```

**Check DNS:**
```bash
nslookup <hostname>
```

**Flush DNS cache:**
```bash
# Windows
ipconfig /flushdns

# Linux
sudo systemd-resolve --flush-caches
```

---
"""

    def _analyze_permission_error(self, query: str) -> str:
        """Intelligent analysis of permission errors"""
        return """## 🔍 Error Analysis: Permission Denied

### 🎯 What This Error Means
Your user account doesn't have permission to access/modify the file or resource.

### 🔧 Root Cause
- File owned by different user (often root)
- Insufficient file permissions
- Directory not writable

### ✅ Solutions

**Option 1: Change file ownership**
```bash
sudo chown $USER:$USER <file-or-directory>
```

**Option 2: Modify permissions**
```bash
chmod 755 <file>  # rwx for owner, rx for others
chmod 644 <file>  # rw for owner, r for others
```

**Option 3: Run with elevated privileges (if appropriate)**
```bash
sudo <command>  # Linux/Mac
# Windows: Run as Administrator
```

### ⚠️ Security Note
Avoid using `chmod 777` or running everything as sudo/admin.

---
"""

    def _analyze_vpn_error(self, query: str) -> str:
        """Intelligent analysis of VPN errors"""
        return """## 🔍 VPN Error Analysis

### 🎯 Analyzing Your VPN Issue

### ✅ Step-by-Step Troubleshooting

**1. Verify Base Internet Connection**
```bash
ping 8.8.8.8
```
If this fails, fix internet first.

**2. Restart VPN Service**
- Close VPN client completely
- Wait 10 seconds
- Reconnect

**3. Try Alternative VPN Server**
- Connect to a different server location
- Some servers may be overloaded

**4. Check Credentials**
- Verify username/password
- Check if account is active/not expired

**5. Flush DNS (if connected but can't browse)**
```bash
# Windows
ipconfig /flushdns

# Linux/Mac
sudo dscacheutil -flushcache
```

**6. Check Firewall/Antivirus**
- Temporarily disable to test
- Allow VPN ports: 443, 500, 1194, 4500

---
"""

    def _analyze_auth_error(self, query: str) -> str:
        """Intelligent analysis of authentication errors"""
        query_lower = query.lower()
        
        if '401' in query_lower:
            return """## 🔍 Error Analysis: 401 Unauthorized

### 🎯 What This Error Means
The request lacks valid authentication credentials.

### 🔧 Root Cause
- Missing or expired API key/token
- Invalid credentials
- Token not included in request headers

### ✅ Solutions

**For API requests:**
```bash
# Check if Authorization header is set
curl -H "Authorization: Bearer <your-token>" <url>
```

**For expired tokens:**
- Refresh your authentication token
- Re-login to get new credentials

---
"""

        if '403' in query_lower:
            return """## 🔍 Error Analysis: 403 Forbidden

### 🎯 What This Error Means
Authentication succeeded, but you don't have permission to access this resource.

### 🔧 Root Cause
- Insufficient permissions/role
- IP address blocked
- Resource requires higher access level

### ✅ Solutions
- Request access from administrator
- Check if your account has the required role
- Verify you're accessing the correct resource

---
"""

        return """## 🔍 Authentication Error Analysis

### 🎯 Common Fixes

**1. Verify Credentials**
- Double-check username/password
- Check for caps lock
- Ensure account is not locked

**2. Reset Password**
- Use "Forgot Password" feature
- Check email for reset link

**3. Clear Browser Cache**
- Clear cookies and cached data
- Try incognito/private mode

**4. Check Token Expiration**
- API tokens may have expired
- Re-authenticate to get new token

---
"""

    def _analyze_ssl_error(self, query: str) -> str:
        """Intelligent analysis of SSL/TLS errors"""
        return """## 🔍 SSL/Certificate Error Analysis

### 🎯 What This Error Means
There's an issue with the SSL/TLS certificate used for secure connections.

### 🔧 Common Causes
- Expired certificate
- Self-signed certificate
- Certificate hostname mismatch
- Missing intermediate certificates

### ✅ Solutions

**1. Check certificate details**
```bash
openssl s_client -connect <host>:443 -servername <host>
```

**2. Update CA certificates**
```bash
# Linux
sudo update-ca-certificates

# Mac
brew install ca-certificates
```

**3. For development (self-signed certs)**
```python
# Python - disable verification (NOT for production!)
import requests
requests.get(url, verify=False)
```

### ⚠️ Security Warning
Never disable SSL verification in production!

---
"""

    def _analyze_windows_error(self, query: str) -> str:
        """Intelligent analysis of Windows errors"""
        import re
        query_lower = query.lower()
        
        # Try to extract error code
        code_match = re.search(r'0x[0-9a-fA-F]+', query)
        error_code = code_match.group(0) if code_match else None
        
        # Common Windows error codes
        windows_errors = {
            '0x80070005': ('Access Denied', 'Run as Administrator or check file permissions'),
            '0x80070057': ('Invalid Parameter', 'Check command syntax or input values'),
            '0x80004005': ('Unspecified Error', 'Try running SFC /scannow'),
            '0x80073712': ('Component Store Corrupted', 'Run DISM /Online /Cleanup-Image /RestoreHealth'),
            '0x800f081f': ('Source Files Not Found', 'Check Windows Update or use installation media'),
        }
        
        if error_code and error_code.lower() in windows_errors:
            name, fix = windows_errors[error_code.lower()]
            return f"""## 🔍 Error Analysis: {error_code} ({name})

### 🎯 What This Error Means
{name}

### ✅ Solution
{fix}

### 🔧 General Windows Repair Steps
```cmd
sfc /scannow
DISM /Online /Cleanup-Image /RestoreHealth
```

---
"""

        return """## 🔍 Windows Error Analysis

### 🎯 General Windows Troubleshooting

**1. System File Checker**
```cmd
sfc /scannow
```

**2. DISM Repair**
```cmd
DISM /Online /Cleanup-Image /RestoreHealth
```

**3. Check Event Viewer**
- Press Win+R, type `eventvwr.msc`
- Check Windows Logs > Application

**4. Update Windows**
- Settings > Update & Security > Windows Update

---
"""

    def _analyze_api_error(self, query: str) -> str:
        """Intelligent analysis of API errors"""
        query_lower = query.lower()
        
        if '500' in query_lower:
            return """## 🔍 Error Analysis: 500 Internal Server Error

### 🎯 What This Error Means
The server encountered an unexpected condition and couldn't complete the request.

### 🔧 Root Cause
- Bug in server-side code
- Database connection failure
- Unhandled exception on server

### ✅ Solutions (If you own the server)
- Check server logs for stack trace
- Verify database connectivity
- Check recent deployments for bugs

### If it's a third-party API
- Wait and retry later
- Check their status page
- Contact their support

---
"""

        if '404' in query_lower:
            return """## 🔍 Error Analysis: 404 Not Found

### 🎯 What This Error Means
The requested resource doesn't exist at the specified URL.

### 🔧 Root Cause
- Incorrect URL/endpoint
- Resource was deleted
- API version mismatch

### ✅ Solutions
- Double-check the URL spelling
- Verify API endpoint in documentation
- Check if resource ID exists
- Use correct API version in URL

---
"""

        return """## 🔍 API Error Analysis

### 🎯 General API Troubleshooting

**1. Check Request Format**
- Verify URL and HTTP method
- Check headers (Content-Type, Authorization)
- Validate request body (JSON format)

**2. Test with curl**
```bash
curl -v -X GET "https://api.example.com/endpoint" \\
  -H "Authorization: Bearer <token>" \\
  -H "Content-Type: application/json"
```

**3. Check API Documentation**
- Verify endpoint spelling
- Check required parameters
- Confirm authentication method

---
"""

    def _analyze_hardware_error(self, query: str) -> str:
        """Intelligent analysis of hardware/memory/RAM issues"""
        query_lower = query.lower()
        
        if 'memory management' in query_lower or ('ram' in query_lower and ('crash' in query_lower or 'install' in query_lower)):
            return f"""## 🔍 Error Analysis: Memory Management / RAM Issue

### 🎯 Analyzing: "{query[:100]}{'...' if len(query) > 100 else ''}"

This is a common issue when new RAM is installed. The **MEMORY_MANAGEMENT** blue screen (BSOD) typically indicates incompatible or faulty memory.

### 🔧 Root Cause Analysis
- **Incompatible RAM** — speed, voltage, or type mismatch with motherboard
- **Faulty RAM module** — defective stick from manufacturing
- **Incorrect seating** — RAM not fully clicked into the slot
- **Mixed RAM specs** — different speeds/brands causing conflicts
- **BIOS needs update** — motherboard firmware doesn't support the new module

### ✅ Step-by-Step Solutions

**Step 1: Reseat the RAM**
1. Power off and unplug the PC
2. Open the case and remove the new RAM stick(s)
3. Clean the gold contacts with a dry cloth
4. Firmly reinsert — you should hear a **click** on both sides

**Step 2: Test with One Stick at a Time**
1. Remove all RAM except one new stick
2. Boot and test for stability
3. If it crashes, try the other stick → identifies the faulty module

**Step 3: Run Windows Memory Diagnostic**
```
Win + R → type "mdsched.exe" → Enter → Restart now
```
This will scan for hardware memory errors on reboot.

**Step 4: Check RAM Compatibility**
- Open **CPU-Z** or check your motherboard manual
- Verify: DDR generation (DDR4/DDR5), Speed (MHz), Voltage
- Ensure total RAM doesn't exceed motherboard maximum

**Step 5: Update BIOS**
- Visit your motherboard manufacturer's website
- Download and install the latest BIOS update
- This often adds support for newer RAM modules

**Step 6: Run memtest86+**
```
Download from: https://www.memtest86.com/
Boot from USB → Run full test (takes ~1 hour)
```
If errors appear → the RAM stick is defective, request RMA/replacement.

### ⚠️ If Issue Persists
- Try using only the **old RAM** to confirm the new sticks are the problem
- Check if XMP/DOCP profile is enabled in BIOS — try disabling it
- Contact the RAM manufacturer for warranty replacement

---

_Would you like me to **create a support ticket** for this hardware issue?_
"""
        
        # Generic memory/crash
        return f"""## 🔍 Error Analysis: System Crash / Memory Issue

### 🎯 Analyzing: "{query[:100]}{'...' if len(query) > 100 else ''}"

### 🔧 Quick Diagnostics

**1. Run Memory Diagnostic**
```
Win + R → mdsched.exe → Restart and check
```

**2. Check Event Viewer for Details**
```
Win + R → eventvwr.msc → Windows Logs → System
```
Look for Critical/Error events around the crash time.

**3. Update Drivers**
- Graphics, chipset, and storage drivers are common culprits
- Use Device Manager → right-click device → Update driver

**4. Check Disk Health**
```
Open CMD as Admin → chkdsk C: /f /r
```

**5. Check for Overheating**
- Use HWMonitor or Core Temp
- CPU above 90°C or GPU above 85°C indicates cooling problems

### 💡 Provide More Details
For a more specific diagnosis, please share:
- Exact error message or blue screen code
- When does it crash? (startup, under load, random)
- Any recent hardware or software changes

---

_Share any extra details if you want a more specific diagnosis._
"""

    def _analyze_generic_error(self, query: str) -> str:
        """Intelligent analysis when no specific error pattern matched"""
        return f"""## 🔍 Error Analysis

### 🎯 Analyzing: "{query[:100]}{'...' if len(query) > 100 else ''}"

I'll search for specific solutions based on your error. Here's a general approach:

### 🔧 Immediate Steps

**1. Read the Full Error Message**
- Note the exact error text
- Look for file paths, line numbers, or error codes

**2. Check Recent Changes**
- What did you change before the error started?
- Any new installations or updates?

**3. Search for the Exact Error**
- Copy the error message and search online
- Check official documentation

### 💡 Pro Tips
- Include the **exact error message** for more specific help
- Mention the **technology/tool** (Git, Docker, Python, etc.)
- Describe what you were **trying to do**

---
"""

    def _get_helpful_response(self, user_query: str) -> str:
        """Generate a helpful response when no tickets are found based on query context"""
        query_lower = user_query.lower()
        
        # VPN issues
        if 'vpn' in query_lower:
            return """No existing tickets found for VPN issues.

**Here are some common VPN troubleshooting steps:**
1. 🔄 **Restart the VPN client** - Close and reopen the application
2. 🌐 **Check your internet connection** - Ensure you have stable connectivity
3. 🔌 **Try a different server** - Switch to another VPN server location
4. 🔑 **Verify credentials** - Make sure your login details are correct
5. 🛡️ **Check firewall settings** - Ensure VPN ports aren't blocked
"""

        # Password issues
        elif 'password' in query_lower or 'reset' in query_lower or 'login' in query_lower:
            return """No existing tickets found for this issue.

**To reset your password at Tandion:**
1. 🔗 Go to the login page and click **"Forgot Password"**
2. 📧 Enter your registered email address
3. 📬 Check your email for the password reset link
4. 🔐 Create a new password (must include uppercase, lowercase, number, and symbol)
"""

        # Email issues
        elif 'email' in query_lower or 'outlook' in query_lower or 'mail' in query_lower:
            return """No existing tickets found for email issues.

**Common email troubleshooting steps:**
1. 🔄 **Restart Outlook** - Close and reopen the application
2. 🌐 **Check internet connection** - Ensure you're connected
3. 🔑 **Re-enter credentials** - Sometimes reauthentication helps
4. 📱 **Try webmail** - Access email via browser at mail.tandion.com
"""

        # Network/connectivity issues  
        elif 'network' in query_lower or 'internet' in query_lower or 'connection' in query_lower or 'wifi' in query_lower:
            return """No existing tickets found for network issues.

**Network troubleshooting steps:**
1. 🔄 **Restart your computer** - Often resolves temporary issues
2. 📶 **Check WiFi connection** - Ensure you're connected to the right network
3. 🔌 **Try a wired connection** - If possible, use ethernet
4. 🔃 **Restart router/modem** - Power cycle your network equipment
"""

        # Software/application issues
        elif 'install' in query_lower or 'software' in query_lower or 'application' in query_lower or 'app' in query_lower:
            return """No existing tickets found for this software issue.

**For software installation/issues:**
1. 🔄 **Restart the application** - Close completely and reopen
2. 🔃 **Update the software** - Check for available updates
3. 🔁 **Reinstall** - Uninstall and reinstall the application
4. 💻 **Check system requirements** - Ensure your system meets minimum specs
"""

        # Default response
        else:
            return """No tickets found matching your criteria.

**I can help you with:**
- 🔍 **Search tickets** - "Show all open tickets", "Find critical bugs"
- ➕ **Create tickets** - "Create a ticket for [your issue]"
- 📊 **Filter by status** - "Show closed tickets", "In progress issues"
- 👤 **Find by assignee** - "Tickets assigned to [name]"

**Or describe your problem** and I'll try to help with troubleshooting steps!"""

    def _format_final_response(self, tool_results: List[ToolResult], user_query: str, include_ai_response: bool = True) -> str:
        """Format a nice final response based on tool results"""
        response_parts = []
        google_results = []
        stack_results = []
        github_results = []
        other_results = []
        query_lower = user_query.lower()
        
        # Separate results by type
        for result in tool_results:
            if result.success:
                data = result.result
                if isinstance(data, dict) and "results" in data:
                    if result.tool_name == "google_search":
                        google_results = data.get("results", [])
                    elif result.tool_name == "stackexchange_search":
                        stack_results = data.get("results", [])
                    elif result.tool_name == "github_mcp_search":
                        github_results = data.get("results", [])
                else:
                    other_results.append(result)
            else:
                other_results.append(result)
        
        # Check if this is a search response (has search results)
        has_search_results = google_results or stack_results or github_results
        
        # 1. FIRST: Always add AI Troubleshooting Response for technical queries
        if include_ai_response:
            ai_response = self._get_ai_troubleshooting_response(user_query)
            response_parts.append(ai_response)
        
        # 2. SECOND: Add Google/Web Search Results
        if google_results:
            response_parts.append("## 🔍 Web Search Results\n")
            for i, r in enumerate(google_results[:5], 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "")[:200]
                link = r.get("link", "")
                response_parts.append(f"**{i}. {title}**")
                if snippet:
                    response_parts.append(f"> {snippet}")
                if link:
                    response_parts.append(f"🔗 {link}")
                response_parts.append("")
        elif has_search_results:
            response_parts.append("## 🔍 Web Search\n")
            response_parts.append("No direct web results found. See StackOverflow results below.\n")
        
        # 3. THIRD: Add StackExchange Results
        if stack_results:
            response_parts.append("## 📚 StackOverflow Similar Issues\n")
            for i, r in enumerate(stack_results[:5], 1):
                title = r.get("title", "No title")
                link = r.get("link", "")
                score = r.get("score", 0)
                response_parts.append(f"**{i}. {title}**")
                response_parts.append(f"   ⭐ Score: {score}")
                if link:
                    response_parts.append(f"   🔗 {link}")
                response_parts.append("")
        elif has_search_results:
            response_parts.append("## 📚 StackOverflow\n")
            response_parts.append("No similar issues found on StackOverflow.\n")
        
        # 4. FOURTH: Add GitHub Results (if any)
        if github_results:
            response_parts.append("## 🐙 GitHub Related Issues\n")
            for i, r in enumerate(github_results[:5], 1):
                title = r.get("title", "No title")
                link = r.get("link", "")
                state = r.get("state", "unknown")
                emoji = "🟢" if state == "open" else "🔴"
                response_parts.append(f"**{i}. {emoji} {title}** [{state}]")
                if link:
                    response_parts.append(f"   🔗 {link}")
                response_parts.append("")
        
        # 5. Handle other tool results (tickets, etc.)
        for result in other_results:
            if result.success:
                data = result.result
                if isinstance(data, dict):
                    if "tickets" in data:
                        tickets = data.get("tickets", [])
                        if isinstance(tickets, list) and len(tickets) > 0:
                            response_parts.append(f"Found **{len(tickets)} ticket(s)**:\n")
                            response_parts.append("| ID | Title | Status | Priority | Assignee |")
                            response_parts.append("|---|---|---|---|---|")
                            for ticket in tickets:  # Show all tickets
                                if isinstance(ticket, dict):
                                    tid = ticket.get("ticket_id", "N/A")
                                    title = str(ticket.get("title", "N/A"))[:40]
                                    if len(str(ticket.get("title", ""))) > 40:
                                        title += "..."
                                    status = ticket.get("status", "N/A")
                                    priority = ticket.get("priority", "N/A")
                                    assignee = ticket.get("assignee", "Unassigned")
                                    if "@" in str(assignee):
                                        assignee = assignee.split("@")[0]
                                    response_parts.append(f"| {tid} | {title} | {status} | {priority} | {assignee} |")
                        else:
                            response_parts.append(self._get_helpful_response(user_query))
                    elif "ticket" in data and data.get("ticket"):
                        ticket = data["ticket"]
                        if isinstance(ticket, dict):
                            response_parts.append("**Ticket Details:**\n")
                            for key, val in ticket.items():
                                response_parts.append(f"- **{key}:** {val}")
                    elif "message" in data:
                        response_parts.append(data['message'])
                    elif "datetime" in data:
                        response_parts.append(f"Current date and time: **{data['datetime']}** ({data.get('day_of_week', '')})")
                    elif "result" in data:
                        response_parts.append(f"Result: **{data['result']}**")
                    else:
                        response_parts.append(str(data))
            else:
                response_parts.append(f"Error: {result.error}")
        
        return "\n".join(response_parts) if response_parts else "I couldn't find any results for your query."
    
    def _handle_technical_issue(self, user_message: str) -> str:
        """
        Handle technical issues by:
        1. Using the LLM (Gemma-2B) to THINK and analyze the problem
        2. Searching external resources (Google/StackOverflow)
        3. Combining both into a comprehensive response
        """
        response_parts = []
        
        # ---- STEP 1: LLM thinks about the problem ----
        print(f"  🧠 LLM thinking about: {user_message[:80]}...")
        try:
            troubleshoot_prompt = f"""You are a skilled IT support expert at Tandion software company. A user has reported a technical problem. Analyze the issue carefully and provide a helpful troubleshooting response.

**User's Problem:** {user_message}

**Your task:**
1. Identify what the problem likely is
2. Explain the possible root causes
3. Provide step-by-step troubleshooting solutions
4. If relevant, suggest diagnostic commands or tools

Be specific, practical, and helpful. Format your response clearly with numbered steps. Do NOT say you will search — provide your own expert analysis."""

            llm_response = self._generate(troubleshoot_prompt, max_tokens=420)
            
            if llm_response and len(llm_response.strip()) > 20:
                response_parts.append("## 🧠 AI Analysis\n")
                response_parts.append(llm_response.strip())
                response_parts.append("\n---\n")
                print(f"  ✓ LLM generated {len(llm_response)} chars of analysis")
            else:
                print(f"  ⚠ LLM response too short ({len(llm_response)} chars), using built-in analysis")
                builtin = self._get_ai_troubleshooting_response(user_message)
                response_parts.append(builtin)
                response_parts.append("\n---\n")
        except Exception as e:
            print(f"  ⚠ LLM generation failed: {e}, using built-in analysis")
            builtin = self._get_ai_troubleshooting_response(user_message)
            response_parts.append(builtin)
            response_parts.append("\n---\n")
        
        # ---- STEP 2: Search external resources ----
        print(f"  🔍 Searching external resources...")
        auto_results = self._auto_search_for_issue(user_message)
        
        google_results = []
        stack_results = []
        
        for result in auto_results:
            if result.success and result.result and isinstance(result.result, dict):
                items = result.result.get('results', [])
                if items:
                    if result.tool_name == "google_search":
                        google_results = items
                    elif result.tool_name == "stackexchange_search":
                        stack_results = items
        
        # ---- STEP 3: Format search results ----
        if google_results:
            response_parts.append("## 🔍 Web Search Results\n")
            for i, r in enumerate(google_results[:5], 1):
                title = r.get("title", "No title")
                snippet = r.get("snippet", "")[:200]
                link = r.get("link", "")
                response_parts.append(f"**{i}. {title}**")
                if snippet:
                    response_parts.append(f"> {snippet}")
                if link:
                    response_parts.append(f"🔗 {link}")
                response_parts.append("")
        
        if stack_results:
            response_parts.append("## 📚 StackOverflow Related Issues\n")
            for i, r in enumerate(stack_results[:5], 1):
                title = r.get("title", "No title")
                link = r.get("link", "")
                score = r.get("score", 0)
                response_parts.append(f"**{i}. {title}**")
                response_parts.append(f"   ⭐ Score: {score}")
                if link:
                    response_parts.append(f"   🔗 {link}")
                response_parts.append("")
        
        if not google_results and not stack_results:
            response_parts.append("\n_External search unavailable. The analysis above is based on the AI model's knowledge._")
        
        # No ticket creation prompt here; UI handles fix confirmation.
        
        return "\n".join(response_parts)

    def _is_ticket_viewing_request(self, query: str) -> bool:
        """Check if the user wants to view/list/show tickets (NOT create)"""
        query_lower = query.lower().strip()
        
        # Must mention tickets/bugs/issues
        has_ticket_word = bool(re.search(r'\b(ticket|tickets|bug|bugs|issue|issues)\b', query_lower))
        if not has_ticket_word:
            return False
        
        # Must have a viewing intent word
        view_patterns = [
            r'\b(show|list|get|find|search|display|view|fetch|see|check)\b',
            r'\b(all|open|closed|my|recent|latest|high|critical|medium|low|in.?progress)\b',
            r'\bhow many\b', r'\bwhat are\b', r'\bwhich\b',
            r'\btickets? (by|for|with|assigned|from|about)\b',
            r'\bticket\s*#', r'\bticket\s*id\b',
        ]
        
        for pattern in view_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def _handle_ticket_viewing(self, user_message: str) -> str:
        """Handle ticket viewing/listing requests by querying the database directly"""
        query_lower = user_message.lower()
        
        # Determine what to query
        result = None
        query_desc = ""
        
        if 'open' in query_lower:
            result = get_tickets_by_status('open')
            query_desc = "open"
        elif 'closed' in query_lower:
            result = get_tickets_by_status('closed')
            query_desc = "closed"
        elif 'in progress' in query_lower or 'in-progress' in query_lower:
            result = get_tickets_by_status('in_progress')
            query_desc = "in-progress"
        elif 'critical' in query_lower or 'p0' in query_lower:
            result = get_tickets_by_priority('P0 - Critical')
            query_desc = "critical priority"
        elif 'high' in query_lower or 'p1' in query_lower:
            result = get_tickets_by_priority('P1 - High')
            query_desc = "high priority"
        elif 'medium' in query_lower or 'p2' in query_lower:
            result = get_tickets_by_priority('P2 - Medium')
            query_desc = "medium priority"
        elif 'low' in query_lower or 'p3' in query_lower:
            result = get_tickets_by_priority('P3 - Low')
            query_desc = "low priority"
        elif re.search(r'assigned to (\w+)', query_lower):
            match = re.search(r'assigned to (\w+)', query_lower)
            assignee = match.group(1)
            result = get_tickets_by_assignee(assignee)
            query_desc = f"assigned to {assignee}"
        elif re.search(r'ticket\s*#?\s*(\d+)', query_lower):
            match = re.search(r'ticket\s*#?\s*(\d+)', query_lower)
            ticket_id = match.group(1)
            result = get_ticket_by_id(ticket_id)
            query_desc = f"ticket #{ticket_id}"
        else:
            # Default: show all open tickets
            result = get_tickets_by_status('open')
            query_desc = "open"
        
        if result is None:
            return self._get_helpful_response(user_message)
        
        # Format the response (NO Error Analysis template)
        return self._format_ticket_results(result, query_desc)

    def _format_ticket_results(self, result: dict, query_desc: str) -> str:
        """Format ticket query results as a clean table without Error Analysis"""
        response_parts = []
        
        # Handle single ticket lookup
        if 'ticket' in result and result.get('ticket'):
            ticket = result['ticket']
            if isinstance(ticket, dict):
                response_parts.append(f"## 🎫 Ticket Details\n")
                for key, val in ticket.items():
                    response_parts.append(f"- **{key}:** {val}")
                return "\n".join(response_parts)
        
        # Handle ticket list
        tickets = result.get('tickets', [])
        
        if isinstance(tickets, list) and len(tickets) > 0:
            response_parts.append(f"## 📋 {query_desc.title()} Tickets\n")
            response_parts.append(f"Found **{len(tickets)}** {query_desc} ticket(s):\n")
            response_parts.append("| ID | Title | Status | Priority | Assignee |")
            response_parts.append("|---|---|---|---|---|")
            for ticket in tickets:
                if isinstance(ticket, dict):
                    tid = ticket.get('ticket_id', 'N/A')
                    title = str(ticket.get('title', 'N/A'))[:45]
                    if len(str(ticket.get('title', ''))) > 45:
                        title += '...'
                    status = ticket.get('status', 'N/A')
                    priority = ticket.get('priority', 'N/A')
                    assignee = ticket.get('assignee', 'Unassigned')
                    if '@' in str(assignee):
                        assignee = assignee.split('@')[0]
                    response_parts.append(f"| {tid} | {title} | {status} | {priority} | {assignee} |")
            return "\n".join(response_parts)
        
        # No tickets found
        error = result.get('error', '')
        if error:
            return f"⚠️ Database error: {error}\n\nPlease check the database connection and try again."
        
        return f"No **{query_desc}** tickets found.\n\n_Try a different filter like \"Show open tickets\" or \"Show critical tickets\"._"

    def _is_technical_issue_query(self, query: str) -> bool:
        """Check if the query is asking about a technical issue that should trigger automatic search"""
        query_lower = query.lower()
        
        # Direct error indicators (high priority)
        error_indicators = [
            'fatal:', 'error:', 'exception:', 'failed:', 'traceback',
            'errno', 'enoent', 'econnrefused', 'eacces', 'modulenotfound',
            'syntaxerror', 'typeerror', 'keyerror', 'valueerror', 'indexerror',
            '0x', 'exit code', 'stack trace'
        ]
        
        # Issue keywords
        issue_keywords = [
            'not working', 'not connecting', 'error', 'issue', 'problem', 'fail',
            'crash', 'freeze', 'slow', 'broken', 'bug', 'fix', 'help', 'why',
            'cannot', "can't", 'unable', 'stuck', 'timeout', 'refused', 'denied',
            'how can i fix', 'how to fix', 'how do i fix', 'how to solve',
            'already exists', 'not found', 'permission denied', 'access denied'
        ]
        
        # Tech keywords
        tech_keywords = [
            'vpn', 'network', 'internet', 'wifi', 'connection', 'server', 'database',
            'login', 'password', 'authentication', 'ssl', 'certificate', 'api',
            'python', 'java', 'javascript', 'code', 'import', 'module', 'package',
            'install', 'update', 'build', 'deploy', 'docker', 'kubernetes',
            'git', 'npm', 'pip', 'remote', 'origin', 'push', 'pull', 'merge',
            'port', 'bind', 'socket', 'http', 'https', 'request', 'response'
        ]
        
        # High priority: direct error indicators
        if any(indicator in query_lower for indicator in error_indicators):
            return True
        
        has_issue = any(kw in query_lower for kw in issue_keywords)
        has_tech = any(kw in query_lower for kw in tech_keywords)
        
        return has_issue or has_tech

    def _auto_search_for_issue(self, query: str) -> List[ToolResult]:
        """Automatically search for solutions when user has a technical issue"""
        results = []
        focused_query = _build_issue_search_query(query)
        
        # Search Google (DuckDuckGo)
        print(f"  🔍 Auto-searching web for: {focused_query}")
        google_result = self.tool_registry.execute(ToolCall(
            tool_name="google_search",
            arguments={"query": f"{focused_query} troubleshooting", "num_results": 5}
        ))
        results.append(google_result)
        
        # Search StackExchange
        print(f"  📚 Auto-searching StackOverflow for: {focused_query}")
        stack_result = self.tool_registry.execute(ToolCall(
            tool_name="stackexchange_search",
            arguments={"query": focused_query, "site": "stackoverflow", "num": 5}
        ))
        results.append(stack_result)
        
        return results

    def _smart_fallback_router(self, user_message: str) -> Optional[List[ToolCall]]:
        """
        Called ONLY when the LLM fails to produce any tool call.
        Uses intent scoring (not brittle keyword lists) to pick the right tool.
        """
        q = user_message.lower().strip()

        # Score each intent
        close_score = sum([
            3 if re.search(r'\b(close|closed|closing|resolve|resolved|resolving)\b', q) else 0,
            2 if re.search(r'\b(mark|set|change|update)\b.{0,30}\b(status|closed|resolved|done|fixed)\b', q) else 0,
            1 if re.search(r'\b(ticket|issue|bug)\b', q) else 0,
            1 if re.search(r'(#\d+|ticket\s*\d+)', q) else 0,
            2 if re.search(r'\b(i needed to|i want to|please|can you)\b.{0,40}\b(close|resolve)\b', q) else 0,
        ])
        create_score = sum([
            3 if re.search(r'\b(create|report|log|submit|file|raise|open)\b.{0,20}\b(ticket|bug|issue)\b', q) else 0,
            2 if re.search(r'\bnew ticket\b', q) else 0,
        ])
        view_score = sum([
            3 if re.search(r'\b(show|list|display|get|fetch|view)\b.{0,20}\b(ticket|bug|issue)\b', q) else 0,
            2 if re.search(r'\b(open|closed|all|my|critical|high|medium|low|in.?progress)\b.{0,10}\btickets?\b', q) else 0,
            1 if re.search(r'\bhow many\b', q) else 0,
        ])
        tech_score = sum([
            3 if re.search(r'\b(not working|not connecting|error|crash|freeze|broken|failed)\b', q) else 0,
            2 if re.search(r'\b(fix|solve|help|why|how)\b', q) else 0,
            1 if re.search(r'\b(vpn|network|python|java|database|server|api|ssl|git)\b', q) else 0,
        ])

        best = max(close_score, create_score, view_score, tech_score)
        if best == 0:
            return None

        print(f"  🧠 Fallback scores — close:{close_score} create:{create_score} view:{view_score} tech:{tech_score}")

        if close_score == best:
            if re.search(r'\b(close|closed|closing)\b', q):
                new_status = 'Closed'
            elif re.search(r'\b(resolve|resolved|resolving|done|fixed)\b', q):
                new_status = 'Resolved'
            elif re.search(r'\bin.?progress\b', q):
                new_status = 'In Progress'
            else:
                new_status = 'Closed'

            ticket_id = None
            id_match = re.search(r'(?:ticket\s*#?|#)(\d+)', q)
            if id_match:
                ticket_id = id_match.group(1)
            else:
                search_q = re.sub(
                    r'\b(i needed to|i want to|please|can you|close|closed|resolve|resolved|'
                    r'update|change|set|mark|the ticket for|the ticket about|ticket|open|'
                    r'a ticket|this ticket)\b', ' ', q
                ).strip()
                search_q = re.sub(r'\s+', ' ', search_q).strip()
                if search_q:
                    sr = search_tickets(search_q)
                    tickets = sr.get('tickets', [])
                    if tickets:
                        ticket_id = str(tickets[0].get('ticket_id', ''))
                        print(f"  ✓ Resolved ticket ID via search: #{ticket_id}")

            if ticket_id:
                return [ToolCall(tool_name='update_ticket_status',
                                 arguments={'ticket_id': ticket_id, 'status': new_status})]
            return None

        if create_score == best:
            return [ToolCall(tool_name='__create_ticket__', arguments={'query': user_message})]

        if view_score == best:
            if re.search(r'\bclosed\b', q):
                return [ToolCall(tool_name='get_tickets_by_status', arguments={'status': 'Closed'})]
            elif re.search(r'\b(critical|p0)\b', q):
                return [ToolCall(tool_name='get_tickets_by_priority', arguments={'priority': 'P0 - Critical'})]
            elif re.search(r'\b(high|p1)\b', q):
                return [ToolCall(tool_name='get_tickets_by_priority', arguments={'priority': 'P1 - High'})]
            elif re.search(r'\b(medium|p2)\b', q):
                return [ToolCall(tool_name='get_tickets_by_priority', arguments={'priority': 'P2 - Medium'})]
            elif re.search(r'\b(low|p3)\b', q):
                return [ToolCall(tool_name='get_tickets_by_priority', arguments={'priority': 'P3 - Low'})]
            elif re.search(r'(#\d+|ticket\s*#?\s*\d+)', q):
                m = re.search(r'(\d+)', q)
                tid = m.group(1) if m else ''
                return [ToolCall(tool_name='get_ticket_by_id', arguments={'ticket_id': tid})]
            else:
                return [ToolCall(tool_name='get_tickets_by_status', arguments={'status': 'Open'})]

        if tech_score == best:
            return [
                ToolCall(tool_name='google_search',
                         arguments={'query': f"{user_message} solution fix", 'num_results': 5}),
                ToolCall(tool_name='stackexchange_search',
                         arguments={'query': user_message, 'site': 'stackoverflow', 'num': 5}),
            ]

        return None

    def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """
        Process a user message: LLM decides which tool to call based on the
        system prompt examples. Smart fallback router is used only when the
        LLM fails to produce a valid tool call.
        """
        if not self.is_available():
            return "I'm sorry, the local model is not available. Please try again later."

        # Guard: off-topic (not bug/ticket/software related)
        off_topic_response = self._check_off_topic(user_message)
        if off_topic_response:
            return off_topic_response

        # Multi-turn ticket creation state
        ticket_creator = get_ticket_creator()
        if ticket_creator.is_waiting():
            response = ticket_creator.process_response(user_message)
            if response:
                return response

        # ReAct-style intent routing before tool-calling
        intent_data = self._infer_intent(user_message)
        intent = intent_data.get("intent", "unknown")
        slots = intent_data.get("slots", {})

        if intent == "troubleshoot":
            return self._handle_technical_issue(user_message)

        if intent == "create_ticket":
            issue_text = slots.get("issue") or user_message
            return ticket_creator.create_ticket_from_query(issue_text)

        if intent == "update_status":
            ticket_id = slots.get("ticket_id")
            status = slots.get("status")
            if not ticket_id or not status:
                return "Please provide the ticket ID and the new status (Open, In Progress, Resolved, Closed)."
            result = update_ticket_status(ticket_id=str(ticket_id), status=status)
            if result.get("success"):
                return f"✅ **Ticket #{ticket_id}** status updated to **{status}**."
            return f"❌ Failed to update ticket #{ticket_id}: {result.get('error', 'Unknown error')}."

        if intent == "update_priority":
            ticket_id = slots.get("ticket_id")
            priority = slots.get("priority")
            if not ticket_id or not priority:
                return "Please provide the ticket ID and the priority (P0 Critical, P1 High, P2 Medium, P3 Low)."
            result = update_ticket_priority(ticket_id=str(ticket_id), priority=priority)
            if result.get("success"):
                return f"✅ **Ticket #{ticket_id}** priority updated to **{priority}**."
            return f"❌ Failed to update ticket #{ticket_id}: {result.get('error', 'Unknown error')}."

        if intent == "view_tickets":
            if slots.get("ticket_id"):
                result = get_ticket_by_id(str(slots.get("ticket_id")))
                return self._format_ticket_results(result, f"ticket #{slots.get('ticket_id')}")
            if slots.get("status"):
                result = get_tickets_by_status(slots.get("status"))
                return self._format_ticket_results(result, str(slots.get("status")))
            if slots.get("priority"):
                result = get_tickets_by_priority(slots.get("priority"))
                return self._format_ticket_results(result, str(slots.get("priority")))
            if slots.get("assignee"):
                result = get_tickets_by_assignee(str(slots.get("assignee")))
                return self._format_ticket_results(result, f"assigned to {slots.get('assignee')}")
            if slots.get("query"):
                result = search_tickets(str(slots.get("query")))
                return self._format_ticket_results(result, "matching")
            return self._handle_ticket_viewing(user_message)

        if intent == "unknown" and self._is_ticket_viewing_request(user_message):
            return self._handle_ticket_viewing(user_message)

        # Direct ticket viewing requests - no LLM/tool-call required
        if self._is_ticket_viewing_request(user_message):
            return self._handle_ticket_viewing(user_message)

        # Technical issues: return dynamic troubleshooting + searches
        if self._is_technical_issue_query(user_message):
            return self._handle_technical_issue(user_message)

        # ----------------------------------------------------------------
        # Unambiguous fast-path: explicit ticket ID + close/resolve/update action
        # e.g. "close ticket 31", "resolve #5", "mark ticket 10 as done"
        # No LLM needed - these two signals together cannot be misinterpreted.
        # ----------------------------------------------------------------
        q_lower = user_message.lower()
        _action_match = re.search(
            r'\b(close|closed|resolve|resolved|resolving|mark.*?(closed|resolved|done|fixed)|'
            r'set.*?(closed|resolved|done|in progress)|update.*?status)\b',
            q_lower
        )
        _id_match = re.search(r'(?:ticket\s*#?|#)\s*(\d+)', q_lower)
        if _action_match and _id_match:
            _ticket_id = _id_match.group(1)
            if re.search(r'\b(close|closed|closing)\b', q_lower):
                _new_status = 'Closed'
            elif re.search(r'\b(resolve|resolved|resolving|done|fixed)\b', q_lower):
                _new_status = 'Resolved'
            elif re.search(r'\bin.?progress\b', q_lower):
                _new_status = 'In Progress'
            else:
                _new_status = 'Closed'
            print(f"  ⚡ Fast-path: update_ticket_status(ticket_id={_ticket_id}, status={_new_status})")
            result = update_ticket_status(ticket_id=_ticket_id, status=_new_status)
            if result.get('success'):
                return f"✅ **Ticket #{_ticket_id}** has been marked as **{_new_status}** successfully."
            else:
                return f"❌ Failed to update ticket #{_ticket_id}: {result.get('error', 'Unknown error')}."

        # ----------------------------------------------------------------
        # Let the LLM decide what tool to use
        # ----------------------------------------------------------------
        system_prompt = self._build_system_prompt()
        conversation = f"{system_prompt}\n\n**User:** {user_message}\n\n**Assistant:**"

        all_tool_results = []

        for iteration in range(max_iterations):
            response = self._generate(conversation, max_tokens=256)
            print(f"  📝 LLM Response: {response[:200]}...")

            available_tools = self.tool_registry.list_tools()
            tool_calls = ToolCallParser.parse(response, available_tools)
            print(f"  🔍 Parsed {len(tool_calls)} tool calls from LLM")

            # Try harder: tool name present but not formatted correctly
            if not tool_calls:
                for tool_name in available_tools:
                    if tool_name in response:
                        match = re.search(
                            rf'{tool_name}[\s:]*[\{{\(]([^\}}\)]+)[\}}\)]', response
                        )
                        if match:
                            try:
                                args_str = match.group(1)
                                if ':' in args_str and '"' in args_str:
                                    arguments = json.loads('{' + args_str + '}')
                                else:
                                    arguments = {}
                                    for kv in args_str.split(','):
                                        if '=' in kv or ':' in kv:
                                            k, v = re.split(r'[=:]', kv, 1)
                                            arguments[k.strip().strip('"\'').strip()] = \
                                                v.strip().strip('"\'').strip()
                                tool_calls.append(
                                    ToolCall(tool_name=tool_name, arguments=arguments)
                                )
                                print(f"  ✓ Extracted from partial match: {tool_name}")
                                break
                            except Exception:
                                pass

            # LLM produced nothing — use smart fallback router
            if not tool_calls:
                print(f"  ⚠ LLM produced no tool call. Using smart fallback router...")
                fallback_calls = self._smart_fallback_router(user_message)

                if fallback_calls:
                    if fallback_calls[0].tool_name == '__create_ticket__':
                        return ticket_creator.create_ticket_from_query(user_message)

                    for tc in fallback_calls:
                        print(f"  🔧 Fallback executing: {tc.tool_name}({tc.arguments})")
                        result = self.tool_registry.execute(tc)
                        all_tool_results.append(result)

                    return self._format_final_response(all_tool_results, user_message)

                if all_tool_results:
                    return self._format_final_response(all_tool_results, user_message)
                return self._clean_response(response)

            # Execute the tool calls the LLM chose
            for tc in tool_calls:
                if tc.tool_name == 'create_ticket' and not tc.arguments.get('title'):
                    return ticket_creator.create_ticket_from_query(user_message)

                print(f"  🔧 Executing: {tc.tool_name}({tc.arguments})")
                result = self.tool_registry.execute(tc)
                all_tool_results.append(result)
                print(f"  ✓ Tool result: success={result.success}")

            if all_tool_results:
                return self._format_final_response(all_tool_results, user_message)

        if all_tool_results:
            return self._format_final_response(all_tool_results, user_message)
        return "I couldn't process your request. Please try rephrasing your question."
    
    def _check_off_topic(self, query: str) -> str:
        """Check if query is off-topic and return redirect message if so"""
        query_lower = query.lower().strip()
        
        # Check for common bugs/solutions request
        if self._is_common_bugs_request(query_lower):
            return self._get_common_bugs_info()
        
        # Keywords that indicate ON-TOPIC (bug/ticket/software related)
        on_topic_keywords = [
            # Ticket operations
            'ticket', 'bug', 'issue', 'report', 'create', 'open', 'close', 'update',
            'assign', 'priority', 'status', 'resolve', 'fix', 'track',
            # Technical/Software
            'error', 'exception', 'crash', 'fail', 'broken', 'not working', 'problem',
            'debug', 'troubleshoot', 'diagnose', 'install', 'configure', 'setup',
            'code', 'software', 'application', 'app', 'program', 'system', 'server',
            'database', 'api', 'network', 'connection', 'timeout', 'memory', 'cpu',
            'slow', 'lag', 'freeze', 'hang', 'stuck', 'loading', 'performance',
            # Technologies
            'python', 'java', 'javascript', 'node', 'npm', 'pip', 'git', 'docker',
            'windows', 'linux', 'mac', 'browser', 'chrome', 'firefox', 'edge',
            'sql', 'mysql', 'postgres', 'mongodb', 'redis', 'aws', 'azure', 'cloud',
            'vpn', 'ssh', 'ssl', 'certificate', 'authentication', 'login', 'password',
            # Common errors
            'fatal', 'warning', 'permission', 'denied', 'access', 'null', 'undefined',
            'import', 'module', 'package', 'dependency', 'version', 'update',
            # Support
            'help', 'support', 'assist', 'tandion', 'employee', 'it ', 'computer', 'laptop',
            'printer', 'monitor', 'keyboard', 'mouse', 'email', 'outlook', 'teams',
            # Bug news/solutions
            'news', 'latest', 'recent', 'common', 'solution', 'vulnerability', 'security',
            'patch', 'hotfix', 'workaround', 'known issue',
        ]
        
        # Check if query contains any on-topic keyword
        for keyword in on_topic_keywords:
            if keyword in query_lower:
                return None  # On-topic, proceed normally
        
        # Basic greetings are OK
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'thanks', 'thank you']
        if query_lower.strip() in greetings or len(query_lower) < 10:
            return None  # Allow greetings
        
        # Off-topic keywords that definitely shouldn't be answered
        off_topic_keywords = [
            'lyrics', 'song', 'music', 'movie', 'film', 'actor', 'actress', 'celebrity',
            'recipe', 'cook', 'food', 'restaurant', 'weather', 'forecast',
            'sports', 'game score', 'team', 'player', 'match',
            'joke', 'funny', 'story', 'poem', 'write me', 'tell me a',
            'translate', 'language', 'definition', 'meaning of',
            'politics', 'election', 'president',
            'travel', 'vacation', 'hotel', 'flight', 'booking',
            'shopping', 'buy', 'price', 'discount', 'amazon', 'ebay',
            'relationship', 'dating', 'love advice',
            'horoscope', 'zodiac', 'astrology',
        ]
        
        for keyword in off_topic_keywords:
            if keyword in query_lower:
                return self._get_redirect_message()
        
        # If query doesn't match any on-topic keywords and is long enough, it's likely off-topic
        if len(query_lower) > 20:
            return self._get_redirect_message()
        
        return None  # Allow by default for short/unclear queries
    
    def _is_common_bugs_request(self, query: str) -> bool:
        """Check if user is asking for common bugs/solutions"""
        patterns = [
            'common bug', 'common issue', 'common problem', 'common error',
            'show bug', 'show issue', 'bug news', 'bug solution',
            'known bug', 'known issue', 'frequent bug', 'frequent issue',
            'typical bug', 'typical issue', 'popular bug', 'top bug',
            'latest bug', 'recent bug', 'trending bug',
        ]
        return any(p in query for p in patterns)
    
    def _get_common_bugs_info(self) -> str:
        """Return common bugs and their solutions"""
        return """📋 **Common Bugs & Solutions**

---

### 🔴 **Git Errors**
| Error | Solution |
|-------|----------|
| `fatal: remote origin already exists` | `git remote remove origin` then add again |
| `error: failed to push some refs` | `git pull --rebase origin main` then push |
| `fatal: not a git repository` | `git init` or check you're in correct folder |

---

### 🟠 **Python Errors**
| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install <module-name>` |
| `IndentationError` | Fix spacing (use 4 spaces, not tabs) |
| `PermissionError` | Run as admin or check file permissions |

---

### 🟡 **Node.js/npm Errors**
| Error | Solution |
|-------|----------|
| `EACCES permission denied` | `npm config set prefix ~/.npm-global` |
| `npm ERR! code ERESOLVE` | `npm install --legacy-peer-deps` |
| `node: command not found` | Reinstall Node.js or fix PATH |

---

### 🔵 **Network/VPN Issues**
| Issue | Solution |
|-------|----------|
| VPN not connecting | Restart VPN client, check credentials |
| SSL Certificate error | Update certificates, check system time |
| Connection timeout | Check firewall, proxy settings |

---

### 🟣 **Windows Issues**
| Issue | Solution |
|-------|----------|
| PC running slow | Clear temp files, disable startup apps |
| App not responding | Task Manager → End task → Restart |
| Blue screen (BSOD) | Update drivers, run `sfc /scannow` |

---

💡 **Need help with a specific bug?** Just paste the error message and I'll help you fix it!

🎫 **Want to report a bug?** Say "Create ticket for [your issue]"
"""
    
    def _get_redirect_message(self) -> str:
        """Return a polite redirect message for off-topic queries"""
        return """🔧 **I'm Tandion Bug Assistant**

I'm specialized in helping with:
• 🎫 **Ticket Management** - Create, view, update tickets
• 🐛 **Bug Troubleshooting** - Debug errors and issues  
• 💻 **Software Support** - IT and technical problems
• 🔍 **Error Analysis** - Find solutions for error messages

**How can I help you today?**

_Try: "Create a ticket for VPN issue" or "Show common bugs" or paste an error message!_"""

    def _is_ticket_update_request(self, query: str) -> bool:
        """Check if the user wants to update/close/resolve an existing ticket"""
        query_lower = query.lower().strip()

        # Must mention a ticket reference
        has_ticket_ref = bool(
            re.search(r'\b(ticket|issue|bug)\b', query_lower) or
            re.search(r'\bticket\s*#?\s*\d+', query_lower) or
            re.search(r'#\d+', query_lower)
        )
        if not has_ticket_ref:
            return False

        # Must have an update/action intent
        update_patterns = [
            r'\b(close|closed|closing)\b',
            r'\b(resolve|resolved|resolving)\b',
            r'\b(update|change|set|mark)\b.{0,30}\b(status|priority)\b',
            r'\b(status|priority)\b.{0,30}\b(update|change|set|mark)\b',
            r'\bmark.{0,20}(closed|resolved|open|done|fixed)\b',
            r'\bset.{0,20}(closed|resolved|open|in progress|done|fixed)\b',
            r'\b(i needed to|i want to|please|can you).{0,30}(close|resolve|update)\b',
        ]
        for pattern in update_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    def _handle_ticket_update(self, user_message: str) -> str:
        """Handle ticket update/close/resolve requests."""
        query_lower = user_message.lower()

        # --- 1. Determine target status ---
        if re.search(r'\b(close|closed|closing)\b', query_lower):
            new_status = 'Closed'
        elif re.search(r'\b(resolve|resolved|resolving|done|fixed)\b', query_lower):
            new_status = 'Resolved'
        elif re.search(r'\bin.?progress\b', query_lower):
            new_status = 'In Progress'
        elif re.search(r'\b(reopen|re-open|open)\b', query_lower):
            new_status = 'Open'
        else:
            # Try to extract explicit status value
            m = re.search(
                r'status.*?(open|in progress|closed|resolved)',
                query_lower
            )
            new_status = m.group(1).title() if m else 'Closed'

        # --- 2. Find ticket ID ---
        ticket_id = None

        # Direct ID mention: "ticket 25", "ticket #25", "#25"
        id_match = re.search(r'(?:ticket\s*#?|#)(\d+)', query_lower)
        if id_match:
            ticket_id = id_match.group(1)

        # If no direct ID, search by title/description keywords
        if not ticket_id:
            # Strip common action words to get the descriptive part
            search_query = re.sub(
                r'\b(i needed to|i want to|please|can you|close|closed|resolve|resolved|'
                r'update|change|set|mark|the ticket for|the ticket about|ticket|open|'
                r'a ticket|this ticket)\b',
                ' ', query_lower
            ).strip()
            search_query = re.sub(r'\s+', ' ', search_query).strip()

            if search_query:
                print(f"  🔍 No ticket ID found, searching by: '{search_query}'")
                search_result = search_tickets(search_query)
                tickets = search_result.get('tickets', [])

                if tickets:
                    # Pick the best match (first result from vector search)
                    best = tickets[0]
                    ticket_id = str(best.get('ticket_id', ''))
                    title = best.get('title', '')
                    print(f"  ✓ Found ticket #{ticket_id}: {title}")
                else:
                    return (
                        f"I couldn't find a ticket matching your description. "
                        f"Please provide the ticket ID directly, e.g. *close ticket #25*."
                    )

        if not ticket_id:
            return (
                "I couldn't identify which ticket to update. "
                "Please include the ticket ID, e.g. *close ticket #25*."
            )

        # --- 3. Perform the update ---
        result = update_ticket_status(ticket_id=ticket_id, status=new_status)

        if result.get('success'):
            return (
                f"✅ **Ticket #{ticket_id}** has been marked as **{new_status}** successfully."
            )
        else:
            error = result.get('error', 'Unknown error')
            return (
                f"❌ Failed to update ticket #{ticket_id}: {error}. "
                f"Please check the ticket ID and try again."
            )

    def _is_ticket_creation_request(self, query: str) -> bool:
        """Check if the user wants to create a ticket (NOT view/list tickets)"""
        query_lower = query.lower().strip()
        
        # CHECK CREATE PHRASES FIRST (more specific, higher priority)
        simple_create_phrases = [
            'create ticket', 'create a ticket', 'create new ticket',
            'new ticket for', 'new ticket about', 'new ticket because',
            'add ticket', 'add a ticket',
            'open a ticket', 'open new ticket',
            'make ticket', 'make a ticket',
            'report bug', 'report a bug', 'report issue', 'report an issue',
            'log ticket', 'log a ticket', 'log issue', 'log an issue',
            'submit ticket', 'submit a ticket', 'submit issue',
            'file ticket', 'file a ticket', 'file bug', 'file a bug',
            'raise ticket', 'raise a ticket', 'raise issue',
            'i want to create', 'i need to create', 'i want to report',
            'can you create', 'please create', 'help me create',
            'create support ticket',
            'please open a ticket', 'open a ticket for', 'open a ticket because',
        ]
        
        for phrase in simple_create_phrases:
            if phrase in query_lower:
                return True
        
        # Regex patterns for create (checked before view patterns)
        create_patterns = [
            r'\b(create|add|make|submit|report|log|raise|file)\s+(a\s+)?(new\s+)?(ticket|bug|issue)',
            r'\bi want to (create|report|log|file|raise|submit)\b',
            r'\bcan you (create|make|add|log)\s+(a\s+)?(ticket|bug|issue)\b',
            r'\bplease (create|make|add|log|open)\s+(a\s+)?(ticket|bug|issue)\b',
            r'\bopen\s+(a\s+)?(new\s+)?(ticket|bug|issue)\s+(for|about|because|regarding)\b',
        ]
        
        for pattern in create_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # THEN: Exclude queries that are about viewing/listing/showing tickets
        # Use word-boundary regex to avoid matching substrings (e.g. "shows" != "show")
        view_patterns = [
            r'\bshow\b', r'\blist\b', r'\bget\b', r'\bfind\b',
            r'\bsearch\b', r'\bdisplay\b', r'\bview\b', r'\bfetch\b',
            r'\ball tickets\b', r'\bmy tickets\b', r'\bopen tickets\b', r'\bclosed tickets\b',
            r'\bhow many\b', r'\bwhat are\b', r'\bwhich tickets\b', r'\bticket status\b',
            r'\btickets by\b', r'\btickets for\b', r'\btickets with\b', r'\bticket #', r'\bticket id\b',
        ]
        
        for pattern in view_patterns:
            if re.search(pattern, query_lower):
                return False  # This is a VIEW request, not CREATE
        
        return False
    
    def _clean_response(self, response: str) -> str:
        """Clean up the response"""
        # Remove any leftover tool call artifacts
        response = re.sub(r'TOOL_CALL:.*', '', response)
        response = re.sub(r'\{[^}]*"tool"[^}]*\}', '', response)
        
        # Remove extra whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        return response.strip()


# ============================================================================
# Singleton instance
# ============================================================================

_local_llm_with_tools: Optional[LocalLLMWithTools] = None

def get_local_llm_with_tools() -> LocalLLMWithTools:
    """Get or create the local LLM with tools instance"""
    global _local_llm_with_tools
    if _local_llm_with_tools is None:
        _local_llm_with_tools = LocalLLMWithTools()
    return _local_llm_with_tools


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Local LLM with Tool Calling...")
    print("=" * 60)
    
    llm = get_local_llm_with_tools()
    
    if llm.is_available():
        test_queries = [
            "Show me all open tickets",
            "What is 25 * 4?",
            "Hello, who are you?",
            "Find critical priority bugs",
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)
            response = llm.chat(query)
            print(f"🤖 Response: {response}")
            print()
    else:
        print("Model not available. Please download the Gemma model first.")
