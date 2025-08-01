from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_PROVIDER: Optional[str] = Field(default=None, description="LLM provider name")
    LLM_API_KEY: str = Field(..., description="API key for the LLM")
    LLM_MODEL: str = Field(default="gpt-3.5-turbo", description="Model name")
    LLM_BASE_URL: str = Field(
        default="https://api.openai.com/v1", description="Base URL for the LLM API"
    )
    LLM_SYSTEM_PROMPT: str = Field(
        default="You are a helpful AI assistant", description="Default system prompt"
    )
    LLM_MAX_TOKENS: int = Field(
        default=4096, description="Maximum number of tokens for generation"
    )
    LLM_TEMPERATURE: float = Field(default=0.2, description="Sampling temperature")
    LLM_FREQUENCY_PENALTY: float = Field(default=0.0, description="Frequency penalty")
    LLM_PRESENCE_PENALTY: float = Field(default=0.0, description="Presence penalty")

    LOGURU_LEVEL: str = Field(default="INFO", description="Logging level")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
