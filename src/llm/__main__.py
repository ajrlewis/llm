import asyncio
import sys

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from loguru import logger

from llm import LLM, AsyncLLM


def main(prompt: str):
    logger.info(f"{prompt = }")
    llm = LLM()
    response = llm.invoke([HumanMessage(content=prompt)])
    if "error" in response:
        logger.error(response["error"])
        return

    logger.info(response["message"]["content"])
    logger.info(response["token_usage"])


async def async_main(prompt: str):
    logger.info(f"{prompt = }")
    llm = AsyncLLM()
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    if "error" in response:
        logger.error(response["error"])
        return

    logger.info(response["message"]["content"])
    logger.info(response["token_usage"])


if __name__ == "__main__":
    prompt = sys.argv[1]
    # main(prompt)
    asyncio.run(async_main(prompt))
