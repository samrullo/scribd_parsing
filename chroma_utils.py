import pathlib
import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched
from tqdm import tqdm

def build_chroma_collection(
    chroma_path: str,
    collection_name: str,
    embedding_func_name: str,
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
    batch_size: int = 200,
):
    chroma_client = chromadb.PersistentClient(chroma_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name},
    )

    document_indices = list(range(len(documents)))
    document_indices_str = [f"id{idx}" for idx in document_indices]

    for batch in tqdm(batched(document_indices, batch_size)):
        start_idx = batch[0]
        end_idx = batch[-1]
        collection.add(
            ids=document_indices_str[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )
