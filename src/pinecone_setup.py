from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec


def ensure_pinecone_index(
    *,
    api_key: str,
    index_name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    pc = Pinecone(api_key=api_key)
    existing = set(pc.list_indexes().names())
    if index_name in existing:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

