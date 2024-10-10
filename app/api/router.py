from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .routes import chat_endpoint

app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# The chat route
app.post("/chat")(chat_endpoint)
