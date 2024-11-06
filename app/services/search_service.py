from typing import List, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models import SteamGame
from app.services.generate_embeddings_service import EmbeddingService


class SearchService:
    def __init__(self, model_setup: Tuple[str, str]):
        self.embedding_service = EmbeddingService(model_setup)

    async def search_games(self, query: str, top_k: int, db: AsyncSession) -> List[SteamGame]:
        query_embedding_tensor = self.embedding_service.encode_texts([query])
        fine_tuned_query = self.embedding_service.fine_tune_embeddings(query_embedding_tensor)

        stmt = (
            select(SteamGame)
            .order_by(SteamGame.embedding.cosine_distance(fine_tuned_query[0]))
            .limit(top_k)
        )

        result = await db.execute(stmt)
        games = result.scalars().all()

        return games
