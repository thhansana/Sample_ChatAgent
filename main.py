import os
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

app = FastAPI()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

SYSTEM = (
    "You are a government services assistant. You only answer questions about "
    "government services. You never make eligibility or legal decisions. "
    "Tell the user you are an AI."
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": req.message}],
    )
    return {"response": resp.content[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
