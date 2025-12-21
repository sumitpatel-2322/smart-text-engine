from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from inferences.classify import predict_sentiment
from inferences.summarize import summarize_text

from contextlib import asynccontextmanager
from config import PRELOAD_MODELS
from inferences.classify import preload_classifier
from inferences.summarize import preload_summarizer

BASE_DIR=Path(__file__).resolve().parent
@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD_MODELS:
        preload_classifier()
        preload_summarizer()
    yield

app=FastAPI(title="Smart Text Engine", lifespan=lifespan)

templates=Jinja2Templates(directory=BASE_DIR/"templates")
app.mount("/static", 
        StaticFiles(directory="static"),
        name="static")
class AnalyzeRequest(BaseModel):
    text:str
    x: int | None=3
@app.get("/")
def home(request:Request):
    return templates.TemplateResponse(
        "index.html",
        {"request":request}
    )
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    summary=summarize_text(req.text, x=req.x or 3)
    sentiment=predict_sentiment(summary)
    return {
        "sentiment":sentiment,
        "summary":summary
    }