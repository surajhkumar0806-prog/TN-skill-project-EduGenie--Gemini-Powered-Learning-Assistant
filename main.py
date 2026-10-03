import os
from pathlib import Path
from enum import Enum
from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from google import genai

# Import custom modules
import explanation_module
import qna
import quiz_module
import summary_module
import learning_path
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="EduGenie - AI Learning Assistant")

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Gemini requires an API key; OAuth access tokens are not accepted here.
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in the project .env file to use Gemini features.")
if api_key.startswith("ya29."):
    raise RuntimeError(
        "GEMINI_API_KEY is an OAuth access token. Use a Gemini API key from Google AI Studio instead."
    )
gemini_client = genai.Client(api_key=api_key)


class TaskType(str, Enum):
    CONCEPT = "concept"
    QA = "qa"
    SUMMARY = "summary"
    QUIZ = "quiz"
    PATH = "path"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "response": None,
            "prompt": "",
            "task": TaskType.CONCEPT.value,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate_response(
    request: Request, prompt: str = Form(...), task: TaskType = Form(...)
):
    ai_response = ""
    model_used = ""

    try:
        if task == TaskType.CONCEPT:
            ai_response = explanation_module.get_concept_explanation(prompt)
            model_used = "LaMini-Flan-T5-783M (Local CPU)"

        elif task == TaskType.QA:
            ai_response = qna.get_answer(gemini_client, prompt)
            model_used = "Gemini 2.5 Flash (Google Cloud)"

        elif task == TaskType.SUMMARY:
            ai_response = summary_module.summarize_text(gemini_client, prompt)
            model_used = "Gemini 2.5 Flash (Google Cloud)"

        elif task == TaskType.QUIZ:
            ai_response = quiz_module.generate_quiz(gemini_client, prompt)
            model_used = "Gemini 2.5 Flash (Google Cloud)"

        elif task == TaskType.PATH:
            ai_response = learning_path.generate_roadmap(gemini_client, prompt)
            model_used = "Gemini 2.5 Flash (Google Cloud)"

    except Exception as e:
        ai_response = f"Error generating response: {str(e)}"
        model_used = "System Error"

        return templates.TemplateResponse(
        request,
        "index.html",
        {
            "response": ai_response,
            "prompt": prompt,
            "task": task.value,
            "model_used": model_used,
        },
    )