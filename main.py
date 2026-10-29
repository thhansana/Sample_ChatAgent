import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
AGENT_VERSION = os.getenv("AGENT_VERSION", "1.0.0")

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
