import os


def test_build_services_smoke() -> None:
    # Smoke test: building pipelines should not call network.
    os.environ["OPENAI_API_KEY"] = "x"
    os.environ["PINECONE_API_KEY"] = "x"

    from src.rag import build_services

    services = build_services(
        pinecone_index_name="team-chat",
        pinecone_namespace="team_chat",
        embedding_dimension=1536,
        openai_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        top_k=3,
    )

    assert services.document_store is not None
    assert services.indexing_pipeline is not None
    assert services.query_pipeline is not None
    assert services.participant_pipeline is not None
    assert services.session_summary_pipeline is not None

