from fastapi import FastAPI
from app.api.router import router
from app.llm.llm_model import BlueViGptModel
from app.database.database import Database

app = FastAPI()
app.include_router(router)
blueViGpt = BlueViGptModel()
db = Database()

@app.on_event("startup")
async def startup_event():
    app.state.model = blueViGpt
    blueViGpt.load_model()
    db.connect()

@app.on_event("shutdown")
async def shutdown_event():
    db.close()

def main():
    pass

if __name__ == "__main__":
    main()