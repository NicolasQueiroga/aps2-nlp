import asyncio
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.train_model import train_autoencoder
from app.core.config import settings

router = APIRouter(
    prefix="/setup-model",
)

init_lock = asyncio.Lock()


class SetupModelRequest(BaseModel):
    model: str = "MiniLM-L6-v2"


class SetupModelResponse(BaseModel):
    message: str


@router.post("", response_model=SetupModelResponse, summary="Train the needed model")
async def setup_model(
    request: SetupModelRequest, db: AsyncSession = Depends(get_db)
) -> SetupModelResponse:
    """
    Train the needed model for the application.

    Args:
    - request: SetupModelRequest - The model to be trained.
    - db: AsyncSession - The database session.

    Returns:
    - SetupModelResponse - The response message.

    Raises:
    - HTTPException: If the model is already downloaded.
    - HTTPException: If the model is not found.
    """

    if f"{settings.sentence_transformer_models[request.model][0]}.pth" in os.listdir(
        settings.embedding_models_path
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} is already downloaded. Check the available models in the documentation.",
        )
    elif request.model in settings.sentence_transformer_models.keys():
        train_autoencoder(request.model)
    elif request.model not in settings.sentence_transformer_models.keys():
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not found. Check the available models in the documentation.",
        )

    return SetupModelResponse(message=f"Model {request.model} is now downloaded.")
