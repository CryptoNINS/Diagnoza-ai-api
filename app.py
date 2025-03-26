from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import whisper
import base64
import tempfile
import os

app = FastAPI()

os.environ["XDG_CACHE_HOME"] = "/app/.cache"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(request: Request):
    try:
        data = await request.json()
        audio_base64 = data.get("audio_base64")

        if not audio_base64:
            return JSONResponse(content={"error": "Brak danych audio"}, status_code=400)

        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        result = model.transcribe(tmp_path)
        return {"text": result["text"]}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"msg": "Diagnoza API działa z Rendera 👋"}
