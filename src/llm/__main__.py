import asyncio
import sys

from loguru import logger

from llm import AsyncChatClient, ChatClient


def main(prompt: str):
    client = ChatClient()
    response = client.create_chat_completion(prompt)

    if "error" in response:
        logger.error(response["error"])
        return

    logger.info(response["message"]["content"])
    logger.info(response["token_usage"])


async def async_main(prompt: str):
    client = AsyncChatClient()
    response = await client.create_chat_completion(prompt)

    if "error" in response:
        logger.error(response["error"])
        return

    logger.info(response["message"]["content"])
    logger.info(response["token_usage"])


async def async_main_stream(prompt: str):
    client = AsyncChatClient()
    async for chunk in client.stream_chat([{"role": "user", "content": prompt}]):
        logger.info(chunk, end="", flush=True)


if __name__ == "__main__":
    prompt = sys.argv[1]
    # main(prompt)
    asyncio.run(async_main(prompt))
