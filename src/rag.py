from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@dataclass(frozen=True)
class RagServices:
    document_store: PineconeDocumentStore
    indexing_pipeline: Pipeline
    query_pipeline: Pipeline
    participant_pipeline: Pipeline
    session_summary_pipeline: Pipeline


def build_services(
    *,
    pinecone_index_name: str,
    pinecone_namespace: str,
    embedding_dimension: int,
    openai_model: str,
    embedding_model: str,
    top_k: int = 8,
) -> RagServices:
    # Haystack OpenAI components read OPENAI_API_KEY from env.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in environment for Haystack OpenAI components.")

    document_store = PineconeDocumentStore(
        index=pinecone_index_name,
        namespace=pinecone_namespace,
        dimension=embedding_dimension,
    )

    # 1) Indexing pipeline: ONLY embedder (per requirement).
    doc_embedder = OpenAIDocumentEmbedder(model=embedding_model)
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", doc_embedder)

    # 2) Query pipeline: text embedder -> PineconeEmbeddingRetriever
    text_embedder = OpenAITextEmbedder(model=embedding_model)
    retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=top_k)
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", text_embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    llm_participant = OpenAIChatGenerator(model=openai_model)
    llm_summary = OpenAIChatGenerator(model=openai_model)

    # 3a) Participant pipeline (default behavior on @mention):
    # Friendly teammate response: no formal minutes / no "next steps" unless explicitly asked.
    participant_template = [
        ChatMessage.from_system(
            "Ты — дружелюбный и компетентный участник рабочего чата команды. Отвечай на русском. "
            "Твоя цель — быть 'в теме' и помогать: дать мнение, совет, уточняющие вопросы, риски, варианты. "
            "НЕ оформляй ответ как итоговое резюме, НЕ делай протокол, НЕ навязывай сроки/этапы/ответственных, "
            "если пользователь явно не попросил. "
            "Опирайся на последние сообщения (dialogue) и найденные сообщения (documents). "
            "Если упоминаешь факты, по возможности укажи, кто и когда это сказал (из meta)."
        ),
        ChatMessage.from_user(
            """
Контекст последних сообщений:
{{dialogue}}

Найденные релевантные сообщения из базы:
{% for document in documents %}
- {{ document.content }} (meta={{ document.meta }})
{% endfor %}

Сообщение/вопрос пользователя:
{{question}}

Ответь коротко и по делу, как коллега в чате.
"""
        ),
    ]

    participant_prompt = ChatPromptBuilder(
        template=participant_template,
        required_variables=["documents", "dialogue", "question"],
    )
    participant_pipeline = Pipeline()
    participant_pipeline.add_component("text_embedder", OpenAITextEmbedder(model=embedding_model))
    participant_pipeline.add_component(
        "retriever", PineconeEmbeddingRetriever(document_store=document_store, top_k=top_k)
    )
    participant_pipeline.add_component("prompt_builder", participant_prompt)
    participant_pipeline.add_component("llm", llm_participant)
    participant_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    participant_pipeline.connect("retriever.documents", "prompt_builder.documents")
    participant_pipeline.connect("prompt_builder.prompt", "llm.messages")

    # 3b) Session summary pipeline (between /start_listening and /stop_listening finalization):
    summary_template = [
        ChatMessage.from_system(
            "Ты — помощник команды в рабочем чате. Отвечай на русском. "
            "Всегда опирайся на контекст из переписки и на найденные сообщения. "
            "Когда приводишь факт, указывай кто и когда это сказал (из meta)."
        ),
        ChatMessage.from_user(
            """
Текущий диалог (последние сообщения, если переданы):
{{dialogue}}

Найденные релевантные сообщения из базы (documents):
{% for document in documents %}
- content: {{ document.content }}
  meta: {{ document.meta }}
{% endfor %}

Задача:
1) Коротко резюмируй обсуждение.
2) Если это был спор — дай взвешенное мнение и аргументы.
3) Если было принято решение/следующие действия — перечисли их списком, с ответственными (если можно определить) и сроками (если были).
4) Если не хватает данных — задай уточняющие вопросы.

Вопрос пользователя (если есть): {{question}}
"""
        ),
    ]

    prompt_builder = ChatPromptBuilder(
        template=summary_template,
        required_variables=["documents", "dialogue", "question"],
    )
    session_summary_pipeline = Pipeline()
    session_summary_pipeline.add_component("text_embedder", OpenAITextEmbedder(model=embedding_model))
    session_summary_pipeline.add_component(
        "retriever", PineconeEmbeddingRetriever(document_store=document_store, top_k=top_k)
    )
    session_summary_pipeline.add_component("prompt_builder", prompt_builder)
    session_summary_pipeline.add_component("llm", llm_summary)
    session_summary_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    session_summary_pipeline.connect("retriever.documents", "prompt_builder.documents")
    session_summary_pipeline.connect("prompt_builder.prompt", "llm.messages")

    return RagServices(
        document_store=document_store,
        indexing_pipeline=indexing_pipeline,
        query_pipeline=query_pipeline,
        participant_pipeline=participant_pipeline,
        session_summary_pipeline=session_summary_pipeline,
    )


def index_documents(
    services: RagServices,
    docs: list[Document],
    *,
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
) -> int:
    """
    Requirement-aligned behavior:
    - Pipeline only creates embeddings (no writer component).
    - We write embeddings to Pinecone ourselves after pipeline run.
    """
    if not docs:
        return 0
    result = services.indexing_pipeline.run({"embedder": {"documents": docs}})
    docs_with_embeddings = result["embedder"]["documents"]
    services.document_store.write_documents(docs_with_embeddings, policy=duplicate_policy)
    return len(docs_with_embeddings)


def retrieve(
    services: RagServices,
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> list[Document]:
    payload: dict[str, Any] = {"text_embedder": {"text": query}}
    if filters is not None:
        payload["retriever"] = {"filters": filters}
    if top_k is not None:
        payload.setdefault("retriever", {})
        payload["retriever"]["top_k"] = top_k
    res = services.query_pipeline.run(payload)
    return res["retriever"]["documents"]


def participant_reply(
    services: RagServices,
    *,
    question: str,
    dialogue: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "text_embedder": {"text": question if question.strip() else dialogue},
        "prompt_builder": {"question": question, "dialogue": dialogue},
    }
    if filters is not None:
        payload["retriever"] = {"filters": filters}
    if top_k is not None:
        payload.setdefault("retriever", {})
        payload["retriever"]["top_k"] = top_k

    res = services.participant_pipeline.run(payload)
    return res["llm"]["replies"][0].text


def session_summary(
    services: RagServices,
    *,
    question: str,
    dialogue: str,
    filters: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "text_embedder": {"text": question if question.strip() else dialogue},
        "prompt_builder": {"question": question, "dialogue": dialogue},
    }
    if filters is not None:
        payload["retriever"] = {"filters": filters}
    if top_k is not None:
        payload.setdefault("retriever", {})
        payload["retriever"]["top_k"] = top_k

    res = services.session_summary_pipeline.run(payload)
    return res["llm"]["replies"][0].text

