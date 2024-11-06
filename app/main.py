from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import search, embedding_model
from app.core.config import settings, EmbeddingAutoencoder
from app.middleware.logging import LoggingMiddleware
import os
import torch
from sentence_transformers import SentenceTransformer
from app.core.database import engine
from app.database.models import Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.get_sentence_model()
    settings.get_autoencoder()
    yield

app = FastAPI(
    title=settings.app_name,
    description="This API handles search requests for games.",
    version="1.0.0",
    debug=settings.debug_mode,
    lifespan=lifespan,
)


app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, tags=["Game Search"])
app.include_router(embedding_model.router, tags=["Model Setup"])
