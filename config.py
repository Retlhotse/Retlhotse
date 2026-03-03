from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    POSTGRES_URL: str = Field(..., description="Async PostgreSQL URL")

    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()