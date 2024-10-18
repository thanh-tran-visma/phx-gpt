from fastapi import FastAPI
from app.api.router import router
from app.llm.llm_model import BlueViGptModel
from app.middleware.middleware import CustomMiddleware
from app.middleware.cors_config import CORSConfig
from contextlib import asynccontextmanager

# Context manager to handle the lifespan (startup and shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    blueViGpt = BlueViGptModel()
    app.state.model = blueViGpt
    blueViGpt.load_model()  # Load your model
    yield
    # Shutdown event (currently no specific shutdown actions)
    pass

# Create FastAPI app with lifespan context
app = FastAPI(lifespan=lifespan)

# Add custom middleware
app.add_middleware(CustomMiddleware)

# Set up CORS middleware (allowing all origins in this example)
#cors_config = CORSConfig(app)
#cors_config.add_cors_middleware()

# Include your app router
app.include_router(router)
