from fastapi import FastAPI
from app.api.router import router
from app.llm.llm_model import BlueViGptModel

app = FastAPI()
app.include_router(router)
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

if __name__ == "__main__":
    main()