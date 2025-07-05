from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ScanRequest(BaseModel):
    text: str

class ScanResponse(BaseModel):
    verdict: str

def fake_ai_detector(text: str) -> str:
    if "bank" in text.lower():
        return "phishing"
    else:
        return "safe"

@app.post("/scan", response_model=ScanResponse)
async def scan_text(request: ScanRequest):
    result = fake_ai_detector(request.text)
    return {"verdict": result}