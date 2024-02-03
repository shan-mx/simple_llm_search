import json
import re
import threading
import traceback
from typing import Generator

import openai
import requests
import toml
import zhipuai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger

_rag_query_text = """
You are a smart AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.
The language of your answer must be same with the language of the question and contexts. Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.
Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.
Here are the set of contexts:
{context}
Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# Load env secrets from secrets.toml.
with open("secrets.toml") as f:
    envs = toml.load(f)


class RAG:
    def __init__(self):
        """
        Initializes rag configs.
        """
        self.model = envs["LLM_NAME"]
        self.backend = envs["SEARCH_BACKEND"]

        if self.backend != "BING":
            raise RuntimeError("Backend must be BING.")

        self.search_api_key = envs["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
        self.search_function = lambda query: self.search_with_bing(
            query,
            self.search_api_key,
        )

        logger.info(f"Using model {self.model}.")
        logger.info(f"Using search backend {self.backend}.")

    def local_client(self):
        """
        Gets a thread-local client, so in case openai clients are not thread safe,
        each thread will have its own client.
        """
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            if envs["LLM_PROVIDER"] == "openai":
                thread_local.client = openai.OpenAI(
                    api_key=envs["OPENAI_API_KEY"]
                )
            elif envs["LLM_PROVIDER"] == "azure":
                thread_local.client = openai.AzureOpenAI(
                    azure_endpoint=envs["AZURE_ENDPOINT"],
                    api_key=envs["AZURE_API_KEY"],
                    api_version=envs["AZURE_API_VERSION"],
                )
            elif envs["LLM_PROVIDER"] == "zhipuai":
                thread_local.client = zhipuai.ZhipuAI(
                    api_key=envs["ZHIPUAI_API_KEY"]
                )
            return thread_local.client

    def search_with_bing(self, query: str, subscription_key: str):
        """
        Search with bing and return the contexts.
        """
        params = {"q": query, "mkt": envs["BING_MKT"]}
        response = requests.get(
            envs["BING_SEARCH_V7_ENDPOINT"],
            headers={"Ocp-Apim-Subscription-Key": subscription_key},
            params=params,
            timeout=envs["DEFAULT_SEARCH_ENGINE_TIMEOUT"],
        )
        if not response.ok:
            logger.error(f"{response.status_code} {response.text}")
            raise HTTPException(response.status_code, "Search engine error.")
        json_content = response.json()
        try:
            contexts = json_content["webPages"]["value"][:envs["REFERENCE_COUNT"]]
        except KeyError:
            logger.error(f"Error encountered: {json_content}")
            return []
        return contexts

    def return_stream_response(
        self, contexts, llm_response
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly.
        """
        # First, yield the contexts.
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""

    def query_function(
        self,
        query: str,
        search_uuid: str,
    ) -> StreamingResponse:
        """
        Query the search engine and returns the response.

        The query can have the following fields:
            - query: the user query.
            - search_uuid: a uuid that is used to identify search query.
        """
        if not search_uuid or not query:
            raise HTTPException(
                status_code=400, detail="query and search_uuid must be provided."
            )

        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i + 1}]] {c['snippet']}" for i,
                    c in enumerate(contexts)]
            )
        )
        try:
            client = self.local_client()
            logger.info(f"Calling LLM for query: {query}")
            llm_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stream=True,
                temperature=0.9,
            )
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        return StreamingResponse(
            self.return_stream_response(
                contexts, llm_response
            ),
            media_type="text/html",
        )


app = FastAPI()
rag = RAG()


@app.post("/query")
async def query_handler(request: Request):
    data = await request.json()
    query = data.get("query")
    search_uuid = data.get("search_uuid")
    return rag.query_function(query, search_uuid)
