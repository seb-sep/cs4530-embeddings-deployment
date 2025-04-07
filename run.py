import modal
from pathlib import Path
from typing import List, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    texts: List[str]
    is_query: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


app = modal.App("fstackoverflow-embeddings")

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-long"

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "transformers", "torch", "einops"
)


@app.function(image=image, gpu="T4", volumes={str(MODEL_DIR): volume})
@modal.fastapi_endpoint()
async def get_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for a list of texts using Arctic Embeddings model.

    Args:
        request: Embedding request containing texts and configuration

    Returns:
        Embedding response containing list of embeddings

    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_pooling_layer=False,
        safe_serialization=True,
    )
    model.eval()

    texts = request.texts
    # the HF docs reccomend prefixing this specifically to queries
    # https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long#usage
    if request.is_query:
        query_prefix = "Represent this sentence for searching relevant passages: "
        texts = [f"{query_prefix}{text}" for text in texts]

    tokens = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )

    with torch.no_grad():
        embeddings = model(**tokens)[0][:, 0]

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return EmbeddingResponse(embeddings=embeddings.tolist())
