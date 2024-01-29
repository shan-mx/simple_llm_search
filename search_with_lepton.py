import concurrent.futures
import glob
import json
import re
import threading
import requests
import traceback
import toml
import zhipuai
from typing import Generator, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from loguru import logger

import leptonai
from leptonai.kv import KV
from leptonai.photon import Photon, StaticFiles
from leptonai.api.workspace import WorkspaceInfoLocalRecord

################################################################################
# Constant values for the RAG model.
################################################################################

# Search engine related. You don't really need to change this.
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "en-US"

# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 6

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5


# If the user did not provide a query, we will use this default query.
_default_query = "Who said 'live long and prosper'?"

# This is really the most important part of the rag model. It gives instructions
# to the model on how to generate the answer. Of course, different models may
# behave differently, and we haven't tuned the prompt to make it optimal - this
# is left to you, application creators, as an open problem.
_rag_query_text = """
You are a smart AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

The language of your answer must be same with the language of the question and contexts. Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
stop_words = []

stop_words_lepton = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

# Load env secrets from secrets.toml

with open("secrets.toml") as f:
    envs = toml.load(f)


def search_with_bing(query: str, subscription_key: str):
    """
    Search with bing and return the contexts.
    """
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException(response.status_code, "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


class RAG(Photon):
    """
    Retrieval-Augmented Generation Demo from Lepton AI.

    This is a minimal example to show how to build a RAG engine with Lepton AI.
    It uses search engine to obtain results based on user queries, and then uses
    LLM models to generate the answer as well as related questions. The results
    are then stored in a KV so that it can be retrieved later.
    """

    requirement_dependency = [
        "openai",  # for openai client usage.
    ]

    extra_files = glob.glob("ui/**/*", recursive=True)

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
        # First, log in to the workspace.
        leptonai.api.workspace.login()
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

        # An executor to carry out async tasks, such as uploading to KV.
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )

        logger.info(f"Using model {self.model}.")
        logger.info(f"Using search backend {self.backend}.")
        # Create the KV to store the search results.
        logger.info("Creating KV. May take a while for the first time.")
        self.kv = KV(
            envs["KV_NAME"], create_if_not_exists=True, error_if_exists=False
        )

    def _raw_stream_response(
        self, contexts, llm_response
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
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

    def stream_and_upload_to_kv(
        self, contexts, llm_response, search_uuid
    ) -> Generator[str, None, None]:
        """
        Streams the result and uploads to KV.
        """
        # First, stream and yield the results.
        all_yielded_results = []
        for result in self._raw_stream_response(
            contexts, llm_response
        ):
            all_yielded_results.append(result)
            yield result
        # Second, upload to KV. Note that if uploading to KV fails, we will silently
        # ignore it, because we don't want to affect the user experience.
        _ = self.executor.submit(self.kv.put, search_uuid, "".join(all_yielded_results))

    @Photon.handler(method="POST", path="/query")
    def query_function(
        self,
        query: str,
        search_uuid: str,
        generate_related_questions: Optional[bool] = True,
    ) -> StreamingResponse:
        """
        Query the search engine and returns the response.

        The query can have the following fields:
            - query: the user query.
            - search_uuid: a uuid that is used to store or retrieve the search result. If
                the uuid does not exist, generate and write to the kv. If the kv
                fails, we generate regardless, in favor of availability. If the uuid
                exists, return the stored result.
            - generate_related_questions: if set to false, will not generate related
                questions. Otherwise, will depend on the environment variable
                RELATED_QUESTIONS. Default: true.
        """
        # Note that, if uuid exists, we don't check if the stored query is the same
        # as the current query, and simply return the stored result. This is to enable
        # the user to share a searched link to others and have others see the same result.
        if search_uuid:
            try:
                result = self.kv.get(search_uuid)
                def str_to_generator(result: str) -> Generator[str, None, None]:
                    yield result

                return StreamingResponse(str_to_generator(result))
            except KeyError:
                logger.info(f"Key {search_uuid} not found, will generate again.")
            except Exception as e:
                logger.error(
                    f"KV error: {e}\n{traceback.format_exc()}, will generate again."
                )
        else:
            raise HTTPException(status_code=400, detail="search_uuid must be provided.")

        # First, do a search query.
        query = query or _default_query
        # Basic attack protection: remove "[INST]" or "[/INST]" from the query
        query = re.sub(r"\[/?INST]", "", query)
        contexts = self.search_function(query)

        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
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
            self.stream_and_upload_to_kv(
                contexts, llm_response, search_uuid
            ),
            media_type="text/html",
        )

    @Photon.handler(mount=True)
    def ui(self):
        return StaticFiles(directory="ui")

    @Photon.handler(method="GET", path="/")
    def index(self) -> RedirectResponse:
        """
        Redirects "/" to the ui page.
        """
        return RedirectResponse(url="/ui/index.html")


if __name__ == "__main__":
    rag = RAG()
    rag.launch()
