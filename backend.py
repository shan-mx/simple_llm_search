import json
import re
import threading
import requests
import traceback
import toml
import zhipuai
from typing import Generator
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger
from leptonai.photon import Photon


_rag_query_text = """
You are a smart AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.
The language of your answer must be same with the language of the question and contexts. Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.
Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.
Here are the set of contexts:
{context}
Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# A set of stop words to use.
stop_words = []

# Load env secrets from secrets.toml.
with open("secrets.toml") as f:
    envs = toml.load(f)


def search_with_bing(query: str, subscription_key: str):
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


class RAG(Photon):
    """
    Retrieval-Augmented Generation Demo from Lepton AI.

    This is a minimal example to show how to build a RAG engine with Lepton AI.
    It uses search engine to obtain results based on user queries, and then uses
    LLM models to generate the answer as well as related questions.
    """

    requirement_dependency = [
        "openai",  # for openai client usage.
    ]

    # It's just a bunch of api calls, so our own deployment can be made massively
    # concurrent.
    handler_max_concurrency = 16

    def local_client(self):
        """
        Gets a thread-local client, so in case openai clients are not thread safe,
        each thread will have its own client.
        """
        import openai

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

    def init(self):
        """
        Initializes photon configs.
        """

        self.backend = envs["SEARCH_BACKEND"]
        self.model = envs["LLM_NAME"]

        if self.backend == "BING":
            self.search_api_key = envs["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
            self.search_function = lambda query: search_with_bing(
                query,
                self.search_api_key,
            )
        else:
            raise RuntimeError("Backend must be BING.")

        logger.info(f"Using model {self.model}.")
        logger.info(f"Using search backend {self.backend}.")

    def _raw_stream_response(
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

    def return_stream_response(
            self, contexts, llm_response, search_uuid
    ) -> Generator[str, None, None]:
        """
        Streams the result.
        """
        # First, stream and yield the results.
        all_yielded_results = []
        for result in self._raw_stream_response(
                contexts, llm_response
        ):
            all_yielded_results.append(result)
            yield result

    @Photon.handler(method="POST", path="/query")
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
        if not search_uuid or query:
            raise HTTPException(status_code=400, detail="query and search_uuid must be provided.")

        # First, do a search query.
        query = query
        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i + 1}]] {c['snippet']}" for i, c in enumerate(contexts)]
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
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)

        return StreamingResponse(
            self.return_stream_response(
                contexts, llm_response, search_uuid
            ),
            media_type="text/html",
        )


if __name__ == "__main__":
    rag = RAG()
    rag.launch()
