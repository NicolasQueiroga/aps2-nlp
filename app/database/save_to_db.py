import os
from typing import Tuple
import pandas as pd
from sqlalchemy import select
from app.core.logger import logger
from app.database.models import SteamGame
from app.services.generate_embeddings_service import EmbeddingService
from app.core.database import AsyncSession


async def initialize_database(db_session, model_setup: Tuple[str, str]):
    embedding_service = EmbeddingService(model_setup)
    csv_path = "data/processed/steam_data.csv"
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found.")
        raise FileNotFoundError(f"CSV file {csv_path} not found.")

    data = pd.read_csv(csv_path)

    required_columns = [
        "name",
        "categories",
        "user_reviews",
        "date",
        "developer",
        "publisher",
        "price",
        "combined_text",
    ]
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")

    logger.info("Generating embeddings for the dataset...")
    combined_texts = data["combined_text"].tolist()
    sentence_embeddings = embedding_service.encode_texts(combined_texts)

    logger.info("Fine-tuning embeddings with the autoencoder...")
    fine_tuned_embeddings = embedding_service.fine_tune_embeddings(sentence_embeddings)

    logger.info("Preparing SteamGame objects for insertion...")
    objects = []
    for idx, row in data.iterrows():
        game = SteamGame(
            name=row["name"],
            categories=row["categories"],
            user_reviews=row["user_reviews"],
            date=row["date"],
            developer=row["developer"],
            publisher=row["publisher"],
            price=row["price"],
            combined_text=row["combined_text"],
            embedding=fine_tuned_embeddings[idx].tolist(),
        )
        objects.append(game)

    db_session.add_all(objects)
    await db_session.commit()
    logger.info(f"Inserted {len(objects)} records into the database.")


async def change_database_embedding(
    db_session: AsyncSession, model_setup: Tuple[str, str]
):
    embedding_service = EmbeddingService(model_setup)

    stmt = select(SteamGame)
    result = await db_session.execute(stmt)
    games = result.scalars().all()

    if not games:
        logger.info("No SteamGame records found to update.")
        return

    combined_texts = [game.combined_text for game in games]
    sentence_embeddings = embedding_service.encode_texts(combined_texts)
    fine_tuned_embeddings = embedding_service.fine_tune_embeddings(sentence_embeddings)

    for idx, game in enumerate(games):
        game.embedding = fine_tuned_embeddings[idx].tolist()

    await db_session.commit()
    logger.info(f"Updated {len(games)} records in the database with new embeddings.")
