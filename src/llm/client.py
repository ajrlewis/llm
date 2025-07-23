from __future__ import annotations

from loguru import logger
from typing import Optional, AsyncGenerator

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL


class ChatClient:
    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        base_url: str = OPENAI_BASE_URL,
        system_prompt: str = "You are a helpful AI assistant",
    ):
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.model = model
        self.system_prompt = system_prompt
        self._client = OpenAI(api_key=api_key, base_url=base_url)

        self.max_tokens = 4096
        self.temperature = 0.2
        self.frequency_penalty = 0
        self.presence_penalty = 0

    def create_chat_completion(
        self,
        user_message: str,
        context: Optional[list[ChatCompletionMessageParam]] = None,
    ) -> dict:
        context = context or []

        has_system = any(msg.get("role") == "system" for msg in context)
        if not has_system:
            context.insert(0, {"role": "system", "content": self.system_prompt})

        context.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=context,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )

            choice = response.choices[0].message.content.strip()
            usage = response.usage

            return {
                "message": {"role": "assistant", "content": choice},
                "token_usage": {
                    "tokens_input": usage.prompt_tokens,
                    "tokens_output": usage.completion_tokens,
                    "tokens_total": usage.total_tokens,
                },
            }
        except Exception as e:
            logger.exception("Chat completion failed")
            return {"error": str(e)}


class AsyncChatClient:
    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        base_url: str = OPENAI_BASE_URL,
        system_prompt: str = "You are a helpful AI assistant",
    ):
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.model = model
        self.system_prompt = system_prompt
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.max_tokens = 4096
        self.temperature = 0.2
        self.frequency_penalty = 0
        self.presence_penalty = 0

    async def create_chat_completion(
        self,
        user_message: str,
        context: Optional[list[ChatCompletionMessageParam]] = None,
    ) -> dict:
        context = context or []

        has_system = any(msg.get("role") == "system" for msg in context)
        if not has_system:
            context.insert(0, {"role": "system", "content": self.system_prompt})

        context.append({"role": "user", "content": user_message})

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=context,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )

            content = response.choices[0].message.content.strip()
            usage = response.usage

            return {
                "message": {"role": "assistant", "content": content},
                "token_usage": {
                    "tokens_input": usage.prompt_tokens,
                    "tokens_output": usage.completion_tokens,
                    "tokens_total": usage.total_tokens,
                },
            }
        except Exception as e:
            logger.exception("Async chat completion failed")
            return {"error": str(e)}

    async def stream_chat(
        self, messages: list[ChatCompletionMessageParam]
    ) -> AsyncGenerator[str, None]:
        messages = [{"role": "system", "content": self.system_prompt}] + messages
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
