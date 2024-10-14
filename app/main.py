from fastapi import FastAPI
from app.api.router import router
from app.model.model import BlueViGptModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.model = BlueViGptModel()

@app.on_event("shutdown")
async def shutdown_event():
    pass

app.include_router(router)