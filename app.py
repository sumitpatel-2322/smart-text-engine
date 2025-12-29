from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import status
from pydantic import BaseModel,Field
from pathlib import Path
from contextlib import asynccontextmanager
from config import PRELOAD_MODELS
from inferences.classify import predict_sentiment, preload_classifier
from inferences.summarize import summarize_text, preload_summarizer
BASE_DIR = Path(__file__).resolve().parent
@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD_MODELS:
        preload_classifier()
        preload_summarizer()
    yield


app = FastAPI(
    title="Smart Text Engine",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid input",
            "message": "Please enter a meaningful movie review (at least 20 characters)."
        }
    )

templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
class AnalyzeRequest(BaseModel):
    text: str=Field(
        ...,
        min_length=20,
        description="Movie Review text(minimum 20 characters)"
    )
class AnalyzeResponse(BaseModel):
    summary:str
    sentiment:str
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
@app.post("/analyze",response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        summary = summarize_text(req.text)
        sentiment_result = predict_sentiment(summary)

        return {
            "summary": summary,
            "sentiment": sentiment_result["sentiment"]
        }
    except Exception:
        return{
            "summary":req.text.strip(),
            "sentiment":"Neutral"
        }