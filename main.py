import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Injected by Agent Manager when the LLM provider is attached
LLM_URL = os.environ["OPENAI_URL"]
LLM_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(
    base_url=LLM_URL,
    api_key=LLM_KEY,
    default_headers={"API-Key": LLM_KEY},   # gateway expects this header
    timeout=300,                            # local CPU model is slow
)
MODEL = os.getenv("LLM_MODEL", "gemma4:e2b")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0.0")

SYSTEM = (
    "You are a government services assistant. Only answer questions about "
    "government services. Never make eligibility or legal decisions. "
    "Tell the user you are an AI. Keep every answer under 3 sentences."
)

sessions: dict[str, list] = {}   # session_id -> message history (no system msg)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[dict] = None

@app.post("/chat")
def chat(req: ChatRequest):
    history = sessions.setdefault(req.session_id, [])
    history.append({"role": "user", "content": req.message})
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        reasoning_effort="none",
        messages=[{"role": "system", "content": SYSTEM}] + history,
    )
    answer = resp.choices[0].message.content or ""
    history.append({"role": "assistant", "content": answer})
    return {"response": answer, "version": AGENT_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
