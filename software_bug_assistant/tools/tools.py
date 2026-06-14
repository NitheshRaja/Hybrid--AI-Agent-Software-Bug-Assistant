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
# add docstring to this module

import os
from datetime import datetime

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.langchain_tool import LangchainTool
try:
    from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams
except ImportError:
    try:
        from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
        from google.adk.tools.mcp_tool.mcp_toolset import StreamableHTTPConnectionParams
    except ImportError:
        MCPToolset = None
        StreamableHTTPConnectionParams = None
from langchain_community.tools import StackExchangeTool
from langchain_community.utilities import StackExchangeAPIWrapper
from toolbox_core import ToolboxSyncClient

# Load environment variables
load_dotenv()


# ----- Example of a Function tool -----
def get_current_date() -> dict:
    """
    Get the current date in the format YYYY-MM-DD
    """
    return {"current_date": datetime.now().strftime("%Y-%m-%d")}


# ----- Example of a Built-in Tool -----
search_agent = Agent(
    model="gemini-2.5-flash",
    name="search_agent",
    instruction="""
    You're a specialist in Google Search.
    """,
    tools=[google_search],
)

search_tool = AgentTool(search_agent)

# ----- Example of a Third Party Tool (LangChainTool) -----
# Initialize StackExchange tool with error handling for connection timeouts
langchain_tool = None
try:
    # Try to initialize StackExchange tool
    # Note: StackExchangeAPIWrapper may attempt to connect during initialization
    # If connection times out, we'll skip it and continue without it
    stack_exchange_wrapper = StackExchangeAPIWrapper()
    stack_exchange_tool = StackExchangeTool(api_wrapper=stack_exchange_wrapper)
    langchain_tool = LangchainTool(stack_exchange_tool)
except (ConnectionError, TimeoutError, Exception) as e:
    # StackExchange API not available (timeout, network issue, etc.)
    # Continue without it - it's not critical for core functionality
    # Only print warning if it's not a timeout (to avoid spam)
    error_str = str(e).lower()
    if "timeout" not in error_str and "connect" not in error_str:
        print(f"Warning: StackExchange tool initialization failed: {e}")
    # Silently continue without StackExchange tool
    langchain_tool = None

# Use local functions instead of the standalone MCP server to avoid Docker dependency issues
from .local_tools import (
    get_tickets_by_status,
    get_tickets_by_priority,
    get_ticket_by_id,
    get_tickets_by_assignee,
    search_tickets,
    create_ticket,
    update_ticket_status,
    update_ticket_priority,
    get_tickets_by_date_range
)

toolbox_tools = [
    get_tickets_by_status,
    get_tickets_by_priority,
    get_ticket_by_id,
    get_tickets_by_assignee,
    search_tickets,
    create_ticket,
    update_ticket_status,
    update_ticket_priority,
    get_tickets_by_date_range
]


# ----- Example of an MCP Tool (streamable-http) -----
# DISABLED: MCP tools cause runtime errors when server is unreachable
# The GitHub MCP server requires a valid GITHUB_PERSONAL_ACCESS_TOKEN
# and causes the entire ADK agent to fail if connection times out.
# Using local tools for GitHub search instead (in local_tools.py)
mcp_tools = None

# Uncomment below if you have a valid GitHub token and want MCP tools:
# github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
# if github_token and len(github_token) > 10:
#     try:
#         mcp_tools = MCPToolset(
#             connection_params=StreamableHTTPConnectionParams(
#                 url="https://api.githubcopilot.com/mcp/",
#                 headers={"Authorization": f"Bearer {github_token}"},
#             ),
#             tool_filter=[
#                 "search_repositories",
#                 "search_issues",
#                 "list_issues",
#                 "get_issue",
#                 "list_pull_requests",
#                 "get_pull_request",
#             ],
#         )
#     except Exception as e:
#         print(f"GitHub MCP tools not available: {e}")
#         mcp_tools = None
