from typing import Optional, AsyncGenerator

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from .config import settings


class LLM:
    def __init__(
        self,
        api_key: str = settings.LLM_API_KEY,
        model: str = settings.LLM_MODEL,
        base_url: str = settings.LLM_BASE_URL,
        system_prompt: str = settings.LLM_SYSTEM_PROMPT,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
        frequency_penalty: float = settings.LLM_FREQUENCY_PENALTY,
        presence_penalty: float = settings.LLM_PRESENCE_PENALTY,
    ):
        if not api_key:
            message = "LLM_API_KEY environment variable is not set"
            logger.error(message)
            raise ValueError(message)

        self.system_message = SystemMessage(content=system_prompt)

        provider = settings.LLM_PROVIDER.upper()

        if provider == "OPENAI":
            self._client = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
        elif provider == "GOOGLE":
            self._client = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            message = f"Unsupported LLM_PROVIDER: {provider}"
            logger.error(message)
            raise ValueError(message)

    def invoke(self, messages: list[BaseMessage] | None = None) -> dict:
        try:
            messages = messages or []

            # Prepend system prompt only if not already present
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages.insert(0, self.system_message)

            logger.debug(f"{messages = }")
            response = self._client.invoke(messages)
            logger.debug(f"{response = }")

            reply = getattr(response, "content", str(response))
            usage = getattr(response, "usage_metadata", {})

            return {
                "message": {"role": "assistant", "content": reply},
                "token_usage": usage,
            }

        except Exception as e:
            logger.exception("Chat completion failed")
            return {"error": str(e)}


class AsyncLLM(LLM):
    async def ainvoke(self, messages: list[BaseMessage] | None = None) -> dict:
        try:
            messages = messages or []

            # Ensure system message is prepended unless already present
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages.insert(0, self.system_message)
            logger.debug(f"{messages = }")

            if hasattr(self._client, "ainvoke"):
                response = await self._client.ainvoke(messages)
            else:
                raise TypeError(
                    f"Client {type(self._client).__name__} does not support async calls"
                )

            logger.debug(f"{response = }")

            reply = getattr(response, "content", str(response))
            usage = getattr(response, "usage_metadata", {})

            return {
                "message": {"role": "assistant", "content": reply},
                "token_usage": usage,
            }

        except Exception as e:
            logger.exception("Async chat completion failed")
            return {"error": str(e)}
