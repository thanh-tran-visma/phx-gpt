from fastapi import FastAPI
from app.api.router import router
from app.llm.llm_model import BlueViGptModel
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()
blueViGpt = BlueViGptModel()

@app.on_event("startup")
async def startup_event():
    app.state.model = blueViGpt
    blueViGpt.load_model()

@app.on_event("shutdown")
async def shutdown_event():
    pass

def main():
    pass

app.include_router(router)

if __name__ == "__main__":
    main()