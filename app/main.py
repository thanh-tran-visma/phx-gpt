from fastapi import FastAPI
from app.api.router import router
from app.llm.llm_model import BlueViGptModel
from app.middleware.middleware import CustomMiddleware
from contextlib import asynccontextmanager


# Context manager to handle the lifespan (startup and shutdown)
@asynccontextmanager
async def lifespan(_app_: FastAPI):
    # Startup event
    yield
    # Shutdown event (currently no specific shutdown actions)
    pass


# Create FastAPI app with lifespan context
app = FastAPI(lifespan=lifespan)

# Add custom middleware
# noinspection PyTypeChecker
app.add_middleware(CustomMiddleware)

# Include your app router
app.include_router(router)
