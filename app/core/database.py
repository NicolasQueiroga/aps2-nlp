from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from sqlalchemy import text

engine = create_async_engine(
    settings.db_url,
    pool_size=20,
    max_overflow=5,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
)

Base = declarative_base()

async def get_db():
    try:
        async with AsyncSessionLocal() as session:
            yield session
    finally:
        await session.close()

