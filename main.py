import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0.0")
import asyncio
import copy
import json
import logging
import os
import re
from typing import Optional

import anthropic
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

log = logging.getLogger("chat-agent")

app = FastAPI()
client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.2.0")

# Injected by Agent Manager when the MCP proxy is attached to this agent (names are configurable there).
MCP_URL = os.getenv("MCP_SERVER_URL")
MCP_KEY = os.getenv("MCP_SERVER_API_KEY")
MCP_AUTH_HEADER = os.getenv("MCP_AUTH_HEADER", "x-api-key")   # must match the proxy's Security tab

# The model only ever sees these tools. The gateway policy is the real control; this is a second belt.
ALLOWED_TOOLS = {"list_agencies", "read_record", "write_field"}
MAX_STEPS = 6

# Tool parameters the agent fills in itself. They are removed from the schema the model sees, so the
# model never chooses or handles them (MCP_SCOPE section 7: the assistant never sees the eID token).
AGENT_SUPPLIED = {"session_id", "assurance_token", "change_id"}

# The demo has no portal, so the citizen pastes the lines printed by `demo_tools.py token` into the chat.
# The agent pulls them out of the message before the model sees it and attaches them to write calls.
APPROVAL_RE = re.compile(r"change_id=(\S+)\s+assurance_token=(\S+)")
APPROVAL_MARK = "[eID approval received]"

SYSTEM = (
    "You are a government services assistant. Only answer questions about "
    "government services. Never make eligibility or legal decisions. "
    "Tell the user you are an AI.\n\n"
    "You can use tools to read residents' records from State agencies and to change one field. "
    "Rules:\n"
    "- Reads need the resident's consent. If a read fails with 'No consent on file', tell the user. "
    "You cannot record consent. Give them this command to run in WSL: "
    "python3 demo_tools.py consent <resident_id> vehicle,elections,tax \"<purpose>\"\n"
    "- Never write without the user's explicit confirmation of the exact change.\n"
    "- A write also needs the citizen's eID approval, which you cannot create and never see. After the user "
    "confirms, give them this command: python3 demo_tools.py token <resident_id> {session_id} --count N "
    "(N = the number of agencies being changed) and ask them to paste the printed lines into the chat. "
    "When they do, the message shows as '" + APPROVAL_MARK + "' and the system attaches one approval to each "
    "write_field call for you. Do not ask for, type or repeat any token.\n"
    "- Each agency needs its own write_field call, and each call uses up one approval.\n"
    "- If write_field says no eID approval is on file, ask the user to run the token command again and paste it.\n"
    "- If a tool returns ok=false, report the agency's reason plainly and do not retry with different values."
)

sessions: dict[str, list] = {}      # session_id -> message history
approvals: dict[str, list] = {}     # session_id -> [(change_id, token)], never shown to the model
_locks: dict[str, asyncio.Lock] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[dict] = None


def _text(content) -> str:
    return "".join(b.text for b in content if getattr(b, "type", "") == "text").strip()


def _redact(args: dict) -> dict:
    return {k: ("***" if k == "assurance_token" else v) for k, v in args.items()}


def _extract_approvals(session_id: str, message: str) -> str:
    """Move eID approvals from the user's message into server-side storage; return the redacted message."""
    found = APPROVAL_RE.findall(message)
    if found:
        approvals.setdefault(session_id, []).extend(found)
    return APPROVAL_RE.sub(APPROVAL_MARK, message)


def _model_schema(schema: dict) -> dict:
    """The schema the model sees: the agent-supplied parameters are removed."""
    out = copy.deepcopy(schema)
    for name in AGENT_SUPPLIED:
        out.get("properties", {}).pop(name, None)
    if "required" in out:
        out["required"] = [r for r in out["required"] if r not in AGENT_SUPPLIED]
    return out


def _parse(body: str) -> dict:
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


@app.get("/health")
def health():
    return {"status": "ok", "mcp_configured": bool(MCP_URL)}


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    if session_id == "default":
        log.warning("No session_id sent: all callers share the 'default' session and eID tokens "
                    "cannot be bound to a real session.")
    async with _locks.setdefault(session_id, asyncio.Lock()):
        return await _chat(req, session_id)


async def _chat(req: ChatRequest, session_id: str):
    history = sessions.setdefault(session_id, [])
    start = len(history)
    history.append({"role": "user", "content": _extract_approvals(session_id, req.message)})
    system = SYSTEM.replace("{session_id}", session_id)
    tool_calls = []

    def unavailable(e: Exception):
        del history[start:]   # drop the half-finished turn so the next message starts clean
        return {"response": f"The assistant is unavailable right now ({type(e).__name__}). Please try again.",
                "version": AGENT_VERSION, "tool_calls": tool_calls}

    # No MCP proxy attached yet: behave like the plain chat agent.
    if not MCP_URL:
        try:
            resp = await client.messages.create(model=MODEL, max_tokens=500, system=system, messages=history)
        except Exception as e:
            return unavailable(e)
        history.append({"role": "assistant", "content": resp.content})
        return {"response": _text(resp.content), "version": AGENT_VERSION, "tool_calls": tool_calls}

    headers = {MCP_AUTH_HEADER: MCP_KEY} if MCP_KEY else {}
    try:
        async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                listed = await mcp.list_tools()
                full_schemas = {t.name: t.inputSchema for t in listed.tools if t.name in ALLOWED_TOOLS}
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": _model_schema(t.inputSchema)}
                         for t in listed.tools if t.name in ALLOWED_TOOLS]
                answer = None
                for _step in range(MAX_STEPS):
                    resp = await client.messages.create(
                        model=MODEL, max_tokens=1000, system=system, tools=tools, messages=history)
                    history.append({"role": "assistant", "content": resp.content})
                    if resp.stop_reason != "tool_use":
                        answer = _text(resp.content)
                        if resp.stop_reason == "max_tokens":
                            answer += "\n\n(The reply was cut off. Ask me to continue.)"
                        break
                    results = []
                    for block in resp.content:
                        if block.type != "tool_use":
                            continue
                        if block.name not in ALLOWED_TOOLS:   # never forward a tool the model was not offered
                            results.append({"type": "tool_result", "tool_use_id": block.id,
                                            "content": "Tool not available", "is_error": True})
                            tool_calls.append({"tool": block.name, "arguments": {}, "result": "blocked by agent"})
                            continue
                        props = full_schemas.get(block.name, {}).get("properties", {})
                        # Whatever the model sent for agent-supplied parameters is discarded.
                        args = {k: v for k, v in dict(block.input).items() if k not in AGENT_SUPPLIED}
                        if "session_id" in props:
                            args["session_id"] = session_id   # always the real session, never model-chosen
                        approval = None
                        if "assurance_token" in props:
                            queue = approvals.get(session_id) or []
                            if not queue:
                                body = json.dumps({"ok": False, "error": "No eID approval on file for this "
                                                   "session. Ask the user to run the token command and paste "
                                                   "the printed lines into the chat."})
                                tool_calls.append({"tool": block.name, "arguments": _redact(args), "result": body})
                                results.append({"type": "tool_result", "tool_use_id": block.id,
                                                "content": body, "is_error": True})
                                continue
                            approval = queue[0]
                            args["change_id"], args["assurance_token"] = approval
                        out = await mcp.call_tool(block.name, args)
                        body = out.content[0].text if out.content else ""
                        if approval is not None:
                            # Spend the approval once the agency has judged it. A refusal that came before the
                            # token check (no consent, field not editable) leaves it usable for a retry.
                            result = _parse(body)
                            if result.get("ok") or "eID confirmation" in str(result.get("error", "")):
                                approvals[session_id].remove(approval)
                        tool_calls.append({"tool": block.name, "arguments": _redact(args), "result": body})
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": body, "is_error": bool(out.isError)})
                    history.append({"role": "user", "content": results})
                if not answer:
                    answer = "I could not finish that request within the allowed number of steps."
        return {"response": answer, "version": AGENT_VERSION, "tool_calls": tool_calls}
    except Exception as e:
        log.exception("chat turn failed")
        del history[start:]
        return {"response": f"The data exchange is unavailable right now ({type(e).__name__}). "
                            "Please try again.", "version": AGENT_VERSION, "tool_calls": tool_calls}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

SYSTEM = (
    "You are a government services assistant. Only answer questions about "
    "government services. Never make eligibility or legal decisions. "
    "Tell the user you are an AI."
)

sessions: dict[str, list] = {}   # session_id -> message history

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[dict] = None

@app.post("/chat")
def chat(req: ChatRequest):
    history = sessions.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})
    resp = client.messages.create(
        model=MODEL, max_tokens=500, system=SYSTEM, messages=history
    )
    answer = resp.content[0].text
    history.append({"role": "assistant", "content": answer})
    return {"response": answer, "version": AGENT_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
