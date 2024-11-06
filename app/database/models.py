from sqlalchemy import Boolean, Column, Integer, String, Index
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.core.config import settings

class SteamGame(Base):
    __tablename__ = "steam_games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, index=True)
    categories = Column(String)
    user_reviews = Column(String)
    date = Column(String)
    developer = Column(String, index=True)
    publisher = Column(String, index=True)
    price = Column(String)
    combined_text = Column(String)
    embedding = Column(Vector())

    __table_args__ = (
        Index("ix_steam_games_embedding", "embedding", postgresql_using="ivfflat"),
    )

class ApiSetup(Base):
    __tablename__ = "api_setup"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String, index=True, nullable=False, default="MiniLM-L6-v2")
    active = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_api_setup_model", "model"),
    )
