import os
import logging
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Literal
from urllib.parse import urljoin, urlparse, parse_qs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
last_search_data: Optional[Any] = None
logger = logging.getLogger(__name__)
from fastapi.responses import HTMLResponse
from openai import OpenAI


APP_TITLE = "MOS Agent"
BASE_URL = "https://support.oracle.com/"
DASHBOARD_PATH = "epmos/faces/Dashboard"
KNOWLEDGE_PATH = "epmos/faces/KMConsolidatedSearch"
PROFILE_DIR = os.getenv("MOS_PROFILE_DIR", "/opt/mos_profile")
FALLBACK_PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".mos_profile")
_profile_dir_in_use = FALLBACK_PROFILE_DIR
os.makedirs(_profile_dir_in_use, exist_ok=True)
PAGE_TIMEOUT_MS = int(os.getenv("MOS_PAGE_TIMEOUT_MS", "30000"))
RESULTS_PER_QUERY_LIMIT = 20
HEADLESS_DEFAULT = os.getenv("MOS_HEADLESS", "1").lower() in {"1", "true", "yes"}
MOS_LOGIN_USER = os.getenv("MOS_LOGIN_USER")
MOS_LOGIN_PASSWORD = os.getenv("MOS_LOGIN_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_mos",
            "description": "Search My Oracle Support for specific error codes or phrases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                        "description": "List of focused MOS search queries.",
                    },
                    "max_per_query": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": RESULTS_PER_QUERY_LIMIT,
                        "default": 5,
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_mos_from_log",
            "description": "Derive MOS searches from a raw log snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_text": {"type": "string"},
                    "max_queries": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 25,
                        "default": 5,
                    },
                    "max_per_query": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": RESULTS_PER_QUERY_LIMIT,
                        "default": 5,
                    },
                },
                "required": ["log_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve the full content of a specific Oracle Support document by its Doc ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The Doc ID of the Oracle Support document to retrieve.",
                    },
                },
                "required": ["doc_id"],
            },
        },
    }
]

# Defensive selectors for global search box (MOS UI can change)
GLOBAL_SEARCH_ARIA = 'aria/Global Search[role="textbox"]'

SEARCH_BOX_SELECTORS = [
    '#pt1\\:svMenu\\:gsb\\:subFgsb\\:mGlobalSearch\\:pt_itG\\:\\:content',
    GLOBAL_SEARCH_ARIA,
    'input[id*="mGlobalSearch" i]',
    'div[role="search"] input',
    'input[aria-label*="Global Search" i]',
    'input[aria-label*="Search" i]',
    'input[placeholder*="Search" i]',
    'input[type="search"]',
    'input[id*="search" i]',
    'input[name*="search" i]',
]

# Potential triggers that reveal or focus the search field
SEARCH_TRIGGER_SELECTORS = [
    'button[aria-label*="Search" i]',
    'a[aria-label*="Search" i]',
    'button:has(svg[aria-label*="search" i])',
    'button:has-text("Search")',
    '#pt1\\:svMenu\\:gsb\\:subFgsb\\:menu_pt_cil2\\:\\:icon',
    'aria/Global Search[role="image"]',
]

SEARCH_SUBMIT_SELECTORS = [
    '#pt1\\:svMenu\\:gsb\\:subFgsb\\:menu_pt_cil2\\:\\:icon',
    'aria/Global Search[role="image"]',
    'button[type="submit"][aria-label*="Search" i]',
]

# Heuristics for detecting login/IDCS screens (avoid overly generic text selectors)
LOGIN_PAGE_HINTS = [
    'form#idcs-signin-basic-signin-form',
    'div[id*="idcs" i]',
    'input[type="password"]',
    'input[name="username"]',
    'input[name="userid"]',
    'text=Oracle Identity Cloud',
]

LOGIN_USERNAME_SELECTORS = [
    'input[name="username"]',
    'input[id*="username" i]',
    'input[type="email"]',
    '#idcs-signin-basic-signin-form input[type="email"]',
]

LOGIN_PASSWORD_SELECTORS = [
    'input[name="password"]',
    'input[id*="password" i]',
    '#idcs-signin-basic-signin-form input[type="password"]',
]

LOGIN_NEXT_SELECTORS = [
    'button[name="signInBtn"]',
    'button:has-text("Next")',
    '#idcs-signin-basic-signin-form button[type="submit"]',
]

LOGIN_SUBMIT_SELECTORS = [
    'button[id*="signin" i]',
    'button[type="submit"]:has-text("Sign In")',
    'button:has-text("Login")',
    '#idcs-signin-basic-signin-form button[type="submit"]',
]


class SearchRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=25)
    max_per_query: int = Field(5, ge=1, le=RESULTS_PER_QUERY_LIMIT)


class LogSearchRequest(BaseModel):
    log_text: str = Field(..., min_length=1)
    max_queries: int = Field(5, ge=1, le=25)
    max_per_query: int = Field(5, ge=1, le=RESULTS_PER_QUERY_LIMIT)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


app = FastAPI(title=APP_TITLE, version="1.0.0")

_play = None
_ctx: Optional[BrowserContext] = None
_ctx_lock = asyncio.Lock()
_ctx_headless: Optional[bool] = None
_openai_client = None



async def _ensure_context(headless: bool) -> BrowserContext:
    global _play, _ctx, _ctx_headless, _profile_dir_in_use
    if _ctx is not None and _ctx_headless == headless:
        return _ctx
    async with _ctx_lock:
        if _ctx is not None and _ctx_headless == headless:
            return _ctx
        # Recreate context with desired head mode
        if _ctx is not None:
            await _ctx.close()
            _ctx = None
        if _play is None:
            _play = await async_playwright().start()
        print(f"[mos_agent] launching Playwright context (headless={headless})")
        launch_error: Optional[Exception] = None
        attempted_paths: List[str] = []
        candidate_paths: List[str] = []
        for cand in (_profile_dir_in_use, PROFILE_DIR, FALLBACK_PROFILE_DIR):
            if cand and cand not in candidate_paths:
                candidate_paths.append(cand)
        for candidate in candidate_paths:
            attempted_paths.append(candidate)
            try:
                os.makedirs(candidate, exist_ok=True)
            except PermissionError:
                print(f"[mos_agent] cannot create profile dir {candidate}: permission denied")


@app.post("/search")
async def search(req: SearchRequest):
    global last_search_data
    result = await _execute_queries(req.queries, req.max_per_query)
    last_search_data = result
    return result
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MOS Agent</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        input, textarea, button { padding: 8px; margin: 5px; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        #results { margin-top: 20px; }
        .result { border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px; background: #f9f9f9; }
        .result:hover { background: #f0f0f0; }
        #chat { height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .message { margin: 5px 0; }
        .user { color: #007bff; }
        .assistant { color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MOS Agent</h1>
        
        <div class="section">
            <h2>Direct MOS Search</h2>
            <input type="text" id="query" placeholder="Enter search terms (e.g., ORA-00600)">
            <input type="number" id="maxPerQuery" value="5" min="1" max="20" title="Max results per query">
            <button onclick="runSearch()">Search</button>
        </div>
        
        <div class="section">
            <h2>Search from Log</h2>
            <textarea id="logText" rows="8" placeholder="Paste error log or stack trace here"></textarea><br>
            <input type="number" id="maxQueries" value="4" min="1" max="25" title="Max queries to generate">
            <input type="number" id="maxPerQueryLog" value="5" min="1" max="20" title="Max results per query">
            <button onclick="runLogSearch()">Search from Log</button>
        </div>
        
        <div class="section">
            <h2>Chat about Results</h2>
            <div id="chat"></div>
            <input type="text" id="message" placeholder="Ask questions about the search results" style="width: 70%;">
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <div id="results"></div>
    </div>
    
    <script>
        let messages = [];
        async function runSearch() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            const maxPer = parseInt(document.getElementById('maxPerQuery').value);
            const resp = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ queries: [query], max_per_query: maxPer })
            });
            const data = await resp.json();
            renderResults(data);
        }
        
        async function runLogSearch() {
            const log = document.getElementById('logText').value.trim();
            if (!log) {
                alert('Please paste log text first');
                return;
            }
            const maxQueries = parseInt(document.getElementById('maxQueries').value);
            const maxPer = parseInt(document.getElementById('maxPerQueryLog').value);
            const resp = await fetch('/search/log', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ log_text: log, max_queries: maxQueries, max_per_query: maxPer })
            });
            const data = await resp.json();
            renderResults(data.results || []);
            if (data.generated_queries && data.generated_queries.length > 0) {
                chatter('assistant', 'Generated queries: ' + data.generated_queries.join(', '));
            }
        }
        
        async function sendMessage() {
            const message = document.getElementById('message').value.trim();
            if (!message) return;
            document.getElementById('message').value = '';
            messages.push({ role: 'user', content: message });
            chatter('user', message);
            const resp = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: messages })
            });
            const data = await resp.json();
            if (data.reply) {
                messages.push({ role: 'assistant', content: data.reply });
                chatter('assistant', data.reply);
            } else {
                chatter('assistant', 'No response');
            }
        }
        
        function renderResults(results) {
            const container = document.getElementById('results');
            container.innerHTML = '<h3>Search Results</h3>';
            if (!results || results.length === 0) {
                container.innerHTML += '<p>No results found</p>';
                return;
            }
            results.forEach((item, index) => {
                const div = document.createElement('div');
                div.className = 'result';
                div.innerHTML = `
                    <strong>${index + 1}. ${item.title || 'No title'}</strong><br/>
                    <small>Doc ID: ${item.doc_id || 'N/A'}</small><br/>
                    <small><a href="${item.url || '#'}" target="_blank">${item.url || 'No URL'}</a></small><br/>
                    <em>${item.snippet || 'No snippet available'}</em>
                `;
                container.appendChild(div);
            });
        }
        
        function chatter(role, content) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<strong>${role === 'user' ? 'You' : 'Assistant'}:</strong> ${content}`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        // Enter key support
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') runSearch();
        });
        document.getElementById('message').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""
