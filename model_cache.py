
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it
    across chunking and vector indexing.
    """

    print("\nLoading embedding model...")

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Embedding model loaded successfully!")

    return model


@lru_cache(maxsize=1)
def get_reranker_model():
    """
    Load the Cross-Encoder model once and reuse it
    for reranking and contextual compression.
    """

    from sentence_transformers import CrossEncoder

    print("\nLoading Cross-Encoder model...")

    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )

    print("Cross-Encoder model loaded successfully!")

    return model
