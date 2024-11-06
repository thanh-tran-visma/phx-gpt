from fastapi import FastAPI
from app.api.router import router
from app.llm.blue_vi_gpt_model import BlueViGptModel
from app.middleware.middleware import CustomMiddleware
from contextlib import asynccontextmanager


# Context manager to handle the lifespan (startup and shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    blue_vi_gpt = BlueViGptModel()
    app.state.model = blue_vi_gpt
    yield
    # Shutdown event (currently no specific shutdown actions)
    pass


# Create FastAPI app with lifespan context
def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    # noinspection PyTypeChecker
    app.add_middleware(CustomMiddleware)
    app.include_router(router)
    return app
