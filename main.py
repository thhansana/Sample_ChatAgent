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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent")

app = FastAPI()
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.8.0")

# How the agent reaches Claude. With USE_LLM_PROVIDER=true every model call goes through the Agent Manager AI
# Gateway (an LLM Service Provider). The gateway holds the real Anthropic key and applies its guardrails (PII
# masking) before anything leaves for Anthropic. Agent Manager injects LLM_PROVIDER_URL (the provider's invoke URL,
# e.g. http://default-default.gateway.localhost:19080/anthropic; the SDK adds /v1/messages) and LLM_PROVIDER_KEY
# (the platform-issued key) when the provider is attached to this agent. Without the flag the agent calls Anthropic
# directly with ANTHROPIC_API_KEY, as before.
USE_LLM_PROVIDER = os.getenv("USE_LLM_PROVIDER", "false").strip().lower() in ("1", "true", "yes", "on")
if USE_LLM_PROVIDER:
    LLM_URL = (os.getenv("LLM_PROVIDER_URL") or "").split(",")[0].strip().rstrip("/")
    LLM_KEY = (os.getenv("LLM_PROVIDER_KEY") or "").strip()
    if not LLM_URL or not LLM_KEY:
        raise RuntimeError("USE_LLM_PROVIDER is true but LLM_PROVIDER_URL or LLM_PROVIDER_KEY is not set. "
                           "Attach the LLM Service Provider to this agent in Agent Manager and redeploy.")
    LLM_AUTH_HEADER = os.getenv("LLM_AUTH_HEADER", "API-Key")   # must match the provider's Security tab
    # The gateway checks the platform key in the API-Key header. It is also sent as the SDK's api_key
    # (x-api-key), as in the console snippet; the gateway replaces that with the real Anthropic key upstream.
    client = anthropic.AsyncAnthropic(api_key=LLM_KEY, base_url=LLM_URL,
                                      default_headers={LLM_AUTH_HEADER: LLM_KEY})
    log.info("LLM: through the AI Gateway at %s (auth header %s)", LLM_URL, LLM_AUTH_HEADER)
else:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    log.info("LLM: direct to Anthropic. No gateway, so no gateway guardrails.")

# Injected by Agent Manager when the MCP proxy is attached to this agent (names are configurable there).
# The console may inject several comma-separated URLs; this agent uses the first one.
MCP_URL = (os.getenv("MCP_SERVER_URL") or "").split(",")[0].strip() or None
MCP_KEY = (os.getenv("MCP_SERVER_API_KEY") or "").strip() or None
MCP_AUTH_HEADER = os.getenv("MCP_AUTH_HEADER", "X-API-Key")   # must match the proxy's Security tab
log.info("MCP proxy URL: %s | API key set: %s", MCP_URL, bool(MCP_KEY))

# The model only ever sees these tools. The gateway policy is the real control; this is a second belt.
ALLOWED_TOOLS = {"list_agencies", "read_record", "write_field"}
MAX_STEPS = 6

# Tool parameters the agent fills in itself. They are removed from the schema the model sees, so the
# model never chooses or handles them (MCP_SCOPE section 7: the assistant never sees the eID token).
AGENT_SUPPLIED = {"session_id", "assurance_token", "change_id"}

# When the portal tells us who is signed in (context.resident_id), the agent also fixes resident_id on every
# tool call and hides it from the model, so the model cannot read or change another resident's records.
# Without a signed-in resident (for example the Agent Manager "Try it" page) the model supplies it as before.
RESIDENT_RE = re.compile(r"[A-Za-z0-9-]{1,32}")
# The note deliberately holds no name or ID: the model is an external service and does not need them
# (the agent fills in resident_id on every call, and the portal greets the resident by name).
RESIDENT_NOTE = (
    "\n\nThe user is a resident who is signed in to the portal. Every tool call already acts on this resident, "
    "so never ask for a resident ID. If the user asks about another person's records or changes, decline and "
    "explain that you can only help with their own."
)

# Approvals reach the agent as 'change_id=... assurance_token=...' in a user message: the portal server sends
# them after the resident approves with their PIN (without the portal, the citizen pastes the lines printed
# by `demo_tools.py token`). The agent pulls them out before the model sees the message and attaches them
# to write calls.
APPROVAL_RE = re.compile(r"change_id=(\S+)\s+assurance_token=(\S+)")
APPROVAL_MARK = "[eID approval received]"

SYSTEM = (
    "You are a government services assistant. Only answer questions about "
    "government services. Never make eligibility or legal decisions. "
    "Tell the user you are an AI.\n\n"
    "You can use tools to read residents' records from State agencies and to change one field. "
    "Rules:\n"
    "- {consent_rule}\n"
    "- Personal details (names, addresses, phone numbers, plates, IDs) appear as placeholders such as "
    "[ADDRESS_1] or [PHONE_1], in the conversation and in tool results. Write a placeholder exactly as given "
    "when you mention the detail or pass it to a tool: the system replaces it with the real value for the "
    "resident. Never guess, invent or change a placeholder, and never mention that placeholders exist.\n"
    "- Always name the agencies as Vehicle Licensing, Elections Office and Revenue Department. The short IDs "
    "(vehicle, elections, tax) are only for tool calls: never say them to the user, and never call a record "
    "the \"tax address\", \"vehicle address\" or \"election address\". Say \"your address at the Revenue "
    "Department\", \"your address at Vehicle Licensing\" or \"your address at the Elections Office\".\n"
    "- Never write without the user's explicit confirmation of the exact change.\n"
    "- A write also needs the citizen's eID approval, which you cannot create and never see. {approval_rule}\n"
    "- Each agency needs its own write_field call, and each call uses up one approval.\n"
    "- If a tool returns ok=false, report the agency's reason plainly and do not retry with different values."
)

# How the user gives the eID approval. In the portal a card with the exact change and a PIN field appears by
# itself when write_field reports 'No eID approval on file'; without the portal the user runs a command.
APPROVAL_PORTAL = (
    "In the portal, the approval card shows the resident the exact change and asks for their eID PIN, so that "
    "card is their confirmation. Once you know the agency, the field and the exact new value, call write_field "
    "straight away; never guess a value. The first write_field call will fail with 'No eID approval on file'. That is expected: "
    "tell the user in one or two sentences that an approval request is now open in the portal (a window asking "
    "for their eID PIN) and that you will make the change once they approve it. Do not tell them to "
    "run any command, and do not retry until a message shows '" + APPROVAL_MARK + "'. When it does, call "
    "write_field again with the same values; the system attaches the approval for you. Do not ask for, type "
    "or repeat any token."
)
APPROVAL_CMD = (
    "After the user confirms, give them this command: python3 demo_tools.py token {resident_id} {session_id} "
    "--count N (N = the number of agencies being changed) and ask them to paste the printed lines into the "
    "chat. When they do, the message shows as '" + APPROVAL_MARK + "' and the system attaches one approval to "
    "each write_field call for you. Do not ask for, type or repeat any token. If write_field says no eID "
    "approval is on file, ask the user to run the token command again and paste it."
)

# Consent is off by default (the resident is already signed in; only changes need the eID approval). Set
# REQUIRE_CONSENT=true here AND on the agency services to bring the consent step back.
REQUIRE_CONSENT = os.getenv("REQUIRE_CONSENT", "false").strip().lower() in ("1", "true", "yes", "on")
CONSENT_NONE = (
    "Reading the signed-in resident's own records needs no consent or other permission, because they have "
    "already signed in: never ask for consent, and never say that consent is required. Only changes need an "
    "approval (below)."
)
# How the user grants consent when it is required. In the portal a consent card appears by itself when a tool
# reports 'No consent on file'; without the portal (Agent Manager "Try it") the user runs a command.
CONSENT_PORTAL = (
    "Reads and writes need the resident's consent. "
    "If a tool fails with 'No consent on file', tell the user in one or two sentences that a consent request "
    "is now open in the portal (a window asking them to approve), and that once they "
    "approve it you will continue. You cannot record consent. Do not tell them to run any command and do "
    "not retry until they say they have approved."
)
CONSENT_CMD = (
    "Reads and writes need the resident's consent. "
    "If a tool fails with 'No consent on file', tell the user. You cannot record consent. Give them this "
    "command to run in WSL: python3 demo_tools.py consent {resident_id} vehicle,elections,tax \"<purpose>\""
)

sessions: dict[str, list] = {}      # session_id -> message history
approvals: dict[str, list] = {}     # session_id -> [(change_id, token)], never shown to the model
residents: dict[str, dict] = {}     # session_id -> {"resident_id", "name"} from the portal
_locks: dict[str, asyncio.Lock] = {}

# Keeping personal details away from the model. The model is an external service, so the agent never sends it a
# resident's details. Known values (the signed-in resident's name and ID, every field of an agency record) are
# swapped for placeholders such as [ADDRESS_1] before anything goes to the model. The real values are put back
# in what the agent returns to the portal and in the tool calls it makes. The map lives only in this process,
# per session. Free text the resident types is only checked against values already known (exact match).
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")
MIN_MASK_LEN = 4                    # shorter known values are not searched for in free text (too many false hits)
vaults: dict[str, dict] = {}        # session_id -> {"real": {value: placeholder}, "ph": {placeholder: value}, "n": {label: count}}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[dict] = None


def _text(content) -> str:
    return "".join(b.text for b in content if getattr(b, "type", "") == "text").strip()


def _root(e: BaseException) -> BaseException:
    """The MCP client wraps errors in ExceptionGroups; dig out the real one."""
    while getattr(e, "exceptions", None):
        e = e.exceptions[0]
    return e


def _redact(args: dict) -> dict:
    return {k: ("***" if k == "assurance_token" else v) for k, v in args.items()}


def _extract_approvals(session_id: str, message: str) -> str:
    """Move eID approvals from the user's message into server-side storage; return the redacted message."""
    found = APPROVAL_RE.findall(message)
    if found:
        approvals.setdefault(session_id, []).extend(found)
    return APPROVAL_RE.sub(APPROVAL_MARK, message)


def _bind_resident(session_id: str, context: Optional[dict]) -> Optional[dict]:
    """The resident this session belongs to, from the portal's context. The first one sticks:
    a later request cannot switch a session to a different resident."""
    bound = residents.get(session_id)
    rid = str((context or {}).get("resident_id") or "").strip()
    if not RESIDENT_RE.fullmatch(rid):
        return bound
    if bound:
        if bound["resident_id"] != rid:
            log.warning("Ignoring resident_id %s for session %s: it is bound to %s", rid, session_id,
                        bound["resident_id"])
        return bound
    name = re.sub(r"[^\w .'-]", "", str((context or {}).get("name") or ""))[:60].strip() or "the resident"
    residents[session_id] = {"resident_id": rid, "name": name}
    log.info("Session %s bound to resident %s", session_id, rid)
    return residents[session_id]


def _placeholder(session_id: str, label: str, value) -> str:
    """The placeholder for a real value in this session (the same value always gets the same one)."""
    v = vaults.setdefault(session_id, {"real": {}, "ph": {}, "n": {}})
    value = str(value)
    if value in v["real"]:
        return v["real"][value]
    label = re.sub(r"[^A-Z0-9]+", "_", str(label).upper()).strip("_") or "VALUE"
    v["n"][label] = v["n"].get(label, 0) + 1
    ph = f"[{label}_{v['n'][label]}]"
    v["real"][value], v["ph"][ph] = ph, value
    return ph


def _mask_text(session_id: str, text: str) -> str:
    """Replace every known real value in the text with its placeholder (whole values, any letter case)."""
    v = vaults.get(session_id)
    values = [r for r in (v["real"] if v else {}) if len(r) >= MIN_MASK_LEN]
    if not values or not text:
        return text
    lookup = {r.lower(): v["real"][r] for r in values}
    pattern = "|".join(re.escape(r) for r in sorted(values, key=len, reverse=True))
    return re.sub(rf"(?<!\w)(?:{pattern})(?!\w)", lambda m: lookup[m.group(0).lower()], text, flags=re.IGNORECASE)


def _mask_obj(session_id: str, obj):
    if isinstance(obj, str):
        return _mask_text(session_id, obj)
    if isinstance(obj, dict):
        return {k: _mask_obj(session_id, x) for k, x in obj.items()}
    if isinstance(obj, list):
        return [_mask_obj(session_id, x) for x in obj]
    return obj


def _mask_result(session_id: str, body: str) -> str:
    """A tool result as the model may see it: every value in a record becomes a placeholder (by field name, so a
    field the agency adds later is masked too), and any other known value is masked wherever it appears."""
    parsed = _parse(body)
    if not parsed:
        return _mask_text(session_id, body)
    if isinstance(parsed.get("record"), dict):
        parsed["record"] = {k: (_placeholder(session_id, k, x) if x not in (None, "") else x)
                            for k, x in parsed["record"].items()}
    return json.dumps(_mask_obj(session_id, parsed), ensure_ascii=False)


def _restore_text(session_id: str, text: str) -> str:
    """Put the real values back for placeholders the model wrote."""
    known = vaults.get(session_id, {}).get("ph", {})

    def put_back(m):
        if m.group(0) in known:
            return known[m.group(0)]
        log.warning("The model used a placeholder that was never issued: %s", m.group(0))
        return m.group(0)
    return PLACEHOLDER_RE.sub(put_back, text)


def _restore_obj(session_id: str, obj):
    if isinstance(obj, str):
        return _restore_text(session_id, obj)
    if isinstance(obj, dict):
        return {k: _restore_obj(session_id, x) for k, x in obj.items()}
    if isinstance(obj, list):
        return [_restore_obj(session_id, x) for x in obj]
    return obj


def _model_schema(schema: dict, hidden: set) -> dict:
    """The schema the model sees: the agent-supplied parameters are removed."""
    out = copy.deepcopy(schema)
    for name in hidden:
        out.get("properties", {}).pop(name, None)
    if "required" in out:
        out["required"] = [r for r in out["required"] if r not in hidden]
    return out


def _parse(body: str) -> dict:
    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {}
    except ValueError:
        return {}


@app.get("/health")
def health():
    return {"status": "ok", "mcp_configured": bool(MCP_URL), "llm_via_gateway": USE_LLM_PROVIDER}


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
    resident = _bind_resident(session_id, req.context)
    if resident:    # known values, masked wherever they appear in what the model sees
        _placeholder(session_id, "RESIDENT_ID", resident["resident_id"])
        if resident["name"] != "the resident":
            _placeholder(session_id, "PERSON", resident["name"])
    history.append({"role": "user", "content": _mask_text(session_id, _extract_approvals(session_id, req.message))})
    hidden = AGENT_SUPPLIED | ({"resident_id"} if resident else set())
    consent_rule = CONSENT_NONE if not REQUIRE_CONSENT else (CONSENT_PORTAL if resident else CONSENT_CMD)
    system = (SYSTEM.replace("{consent_rule}", consent_rule)
              .replace("{approval_rule}", APPROVAL_PORTAL if resident else APPROVAL_CMD)
              .replace("{session_id}", session_id)
              .replace("{resident_id}", "<resident_id>"))   # only used in command-line mode, where there is no signed-in resident
    if resident:
        system += RESIDENT_NOTE
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
        return {"response": _restore_text(session_id, _text(resp.content)), "version": AGENT_VERSION,
                "tool_calls": tool_calls}

    # The console's own snippet sends "API-Key"; the proxy's Security tab shows the header it checks.
    # Send both until it is confirmed which one the gateway uses, then drop the other.
    headers = {MCP_AUTH_HEADER: MCP_KEY, "API-Key": MCP_KEY} if MCP_KEY else {}
    try:
        async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                listed = await mcp.list_tools()
                full_schemas = {t.name: t.inputSchema for t in listed.tools if t.name in ALLOWED_TOOLS}
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": _model_schema(t.inputSchema, hidden)}
                         for t in listed.tools if t.name in ALLOWED_TOOLS]
                answer = None
                for _step in range(MAX_STEPS):
                    resp = await client.messages.create(
                        model=MODEL, max_tokens=1000, system=system, tools=tools, messages=history)
                    history.append({"role": "assistant", "content": resp.content})
                    if resp.stop_reason != "tool_use":
                        answer = _restore_text(session_id, _text(resp.content))
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
                        # Whatever the model sent for agent-supplied parameters is discarded. Placeholders the
                        # model wrote are replaced by the real values before the call leaves the agent.
                        args = {k: v for k, v in _restore_obj(session_id, dict(block.input)).items()
                                if k not in hidden}
                        if "session_id" in props:
                            args["session_id"] = session_id   # always the real session, never model-chosen
                        if resident and "resident_id" in props:
                            args["resident_id"] = resident["resident_id"]   # always the signed-in resident
                        approval = None
                        if "assurance_token" in props:
                            queue = approvals.get(session_id) or []
                            if not queue:
                                # The portal looks for this exact start of the message to raise an approval card.
                                hint = ("An approval request is now waiting for the resident in the portal. "
                                        "Tell them, and wait until they approve it." if resident else
                                        "Ask the user to run the token command and paste the printed lines "
                                        "into the chat.")
                                body = json.dumps({"ok": False, "error": "No eID approval on file for this "
                                                   "session. " + hint})
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
                            if result.get("ok") and args.get("value"):
                                _placeholder(session_id, args.get("field", "VALUE"), args["value"])   # now a known value
                        # The portal gets the real result (it is shown to the resident); the model gets it masked.
                        tool_calls.append({"tool": block.name, "arguments": _redact(args), "result": body})
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": _mask_result(session_id, body), "is_error": bool(out.isError)})
                    history.append({"role": "user", "content": results})
                if not answer:
                    answer = "I could not finish that request within the allowed number of steps."
        return {"response": answer, "version": AGENT_VERSION, "tool_calls": tool_calls}
    except Exception as e:
        del history[start:]
        root = _root(e)
        log.exception("MCP call failed: %r", root)
        return {"response": f"The data exchange is unavailable right now ({type(root).__name__}: "
                            f"{str(root)[:160]}). Please try again.",
                "version": AGENT_VERSION, "tool_calls": tool_calls}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
