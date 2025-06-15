import os
from typing import Dict, Optional
from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

def get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

OPENAI_API_KEY = get_env("OPENAI_API_KEY")
QDRANT_URL = get_env("QDRANT_URL")
QDRANT_INDEX = get_env("QDRANT_INDEX")
QDRANT_EMBEDDING_DIM = int(get_env("QDRANT_EMBEDDING_DIM"))



app = FastAPI()

template = [ChatMessage.from_user("""You are a Kubernetes expert AI assistant. Use only the following technical documentation and logs to answer the user's question accurately.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ question }}

Answer concisely and accurately using Kubernetes best practices.
Prioritize information from the provided Context. If the Context does not contain enough information to answer the question, state that the answer cannot be found in the provided documentation. Do not invent information.
Provide the output in the following format:
Error: {Explain error here}
Solution: {Step by step solution here}
""")]

document_store = QdrantDocumentStore(
    url=QDRANT_URL,
    embedding_dim=QDRANT_EMBEDDING_DIM,
    index=QDRANT_INDEX
)

builder = ChatPromptBuilder(
    template=template,
    required_variables={"documents", "question"}
)

chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = QdrantEmbeddingRetriever(document_store=document_store)

pipe = Pipeline()
pipe.add_component("embedder", embedder)
pipe.add_component("retriever", retriever)
pipe.add_component("chat_prompt_builder", builder)
pipe.add_component("llm", chat_generator)

pipe.connect("embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "chat_prompt_builder.documents")
pipe.connect("chat_prompt_builder.prompt", "llm.messages")


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = {}



class CompletionResponse(BaseModel):
    model: str
    created_at: str
    response: str


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    try:
        print("=== Incoming Prompt ===")
        print(request.prompt)
        print("=======================")

        temperature = float(request.options.get("temperature", "0.7"))
        max_tokens = int(request.options.get("max_tokens", "2048"))
        print(f"Temperature: {temperature}, Max Tokens: {max_tokens}")

        result = pipe.run(
            data={
                "embedder": {"text": request.prompt},
                "chat_prompt_builder": {"question": request.prompt},
                "retriever": {"top_k": 5},
                "llm": {
                    "generation_kwargs": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                }
            }
        )
        llm_reply = result["llm"]["replies"][0]
        response_text = str(llm_reply.text) if hasattr(llm_reply, "text") else str(llm_reply)

        return CompletionResponse(
            model=request.model,
            created_at=datetime.utcnow().isoformat() + "Z",
            response=response_text
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

