import asyncio
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logger import logger
from app.database.models import ApiSetup, SteamGame
from app.database.save_to_db import change_database_embedding, initialize_database
from app.services.search_service import SearchService
from app.core.config import settings

router = APIRouter(
    prefix="/search",
)

init_lock = asyncio.Lock()


class SteamGameResponse(BaseModel):
    id: int
    name: str
    categories: str
    user_reviews: str
    date: str
    developer: str
    publisher: str
    price: str


async def deactivate_all_models_except(db: AsyncSession, model: str):
    stmt = select(ApiSetup).where(ApiSetup.model != model)
    result = await db.execute(stmt)
    api_setups = result.scalars().all()

    for api_setup in api_setups:
        if api_setup.active:
            api_setup.active = False

    await db.commit()


async def activate_model(db: AsyncSession, model: str):
    stmt = select(ApiSetup).where(ApiSetup.model == model)
    result = await db.execute(stmt)
    api_setup = result.scalars().first()

    if api_setup is None:
        new_setup = ApiSetup(model=model, active=True)
        db.add(new_setup)
        await db.commit()
    elif not api_setup.active:
        api_setup.active = True
        await db.commit()


@router.get(
    "", response_model=List[SteamGameResponse], summary="Search for Steam Games"
)
async def search_games(
    query: str = Query(
        ...,
        example="action adventure game",
        description="The search term to look for in the game database.",
    ),
    top_k: int = Query(
        10, example=5, description="The number of top results to return."
    ),
    model_name: str = Query(
        "MiniLM-L6-v2",
        example="MiniLM-L6-v2",
        description="The model to use for generating embeddings. Check the available models in the documentation.",
    ),
    db: AsyncSession = Depends(get_db),
) -> List[SteamGameResponse]:
    """
    Search for Steam games based on a query string.

    This endpoint performs the following actions:
    1. **Database Initialization**: Checks if the database is initialized. If empty, it initializes the database by loading data and generating embeddings.
    2. **Embedding Generation**: Generates embeddings for the search query using the pre-trained models.
    3. **Similarity Search**: Searches the database for the top similar games based on the query embeddings.
    4. **Response**: Returns a list of matching games with their details.

    **Parameters:**
    - `query` (str): The search term to look for in the game database.
    - `top_k` (int): The number of top results to return.
    - `model` (str): The model to use for generating embeddings. Check the available models in the documentation.
    - `db` (AsyncSession): The database session provided by dependency injection.

    **Returns:**
    - `List[SteamGameResponse]`: A list of Steam games that match the search criteria.
    """
    if model_name not in settings.sentence_transformer_models.keys():
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} not found. Check the available models in the documentation.",
        )

    if f"{settings.sentence_transformer_models[model_name][0]}.pth" not in os.listdir(
        settings.embedding_models_path
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model {settings.sentence_transformer_models[model_name][0]}.pth not found. Download it via the /setup-model endpoint.",
        )

    async with init_lock:
        count_stmt = select(func.count(SteamGame.id))
        count_result = await db.execute(count_stmt)
        total = count_result.scalar()

        if total == 0:
            logger.info(f"Database is empty. Initializing with model {model_name}...")
            await initialize_database(
                db, settings.sentence_transformer_models[model_name]
            )
            await deactivate_all_models_except(db, model_name)
            await activate_model(db, model_name)
            logger.info("Database initialization complete.")
        else:
            stmt = select(ApiSetup).where(ApiSetup.model == model_name)
            result = await db.execute(stmt)
            api_setup = result.scalars().first()

            if not api_setup or not api_setup.active:
                await change_database_embedding(
                    db, settings.sentence_transformer_models[model_name]
                )
                await deactivate_all_models_except(db, model_name)
                await activate_model(db, model_name)

    search_service = SearchService(settings.sentence_transformer_models[model_name])
    games = await search_service.search_games(query, top_k, db)

    if not games:
        raise HTTPException(status_code=404, detail="No games found.")

    return games
