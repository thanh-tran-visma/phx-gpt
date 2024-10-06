from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routes import chat_endpoint

app = FastAPI()

# Serve static files
app.mount("/src", StaticFiles(directory="src"), name="static")
templates = Jinja2Templates(directory="src")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"key": "value"})

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# The chat route
app.post("/chat")(chat_endpoint)
