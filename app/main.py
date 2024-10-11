from fastapi import FastAPI
from app.api import app
from app.model.model import BlueViGptModel
from dotenv import load_dotenv

load_dotenv()
blueViGpt = BlueViGptModel()

@app.on_event("startup")
async def startup_event():
    app.state.model = blueViGpt

@app.on_event("shutdown")
async def shutdown_event():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
