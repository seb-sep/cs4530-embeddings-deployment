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
@modal.fastapi_endpoint(
    method="POST"
)  # use POST because we pass a lot of data in request body
async def get_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for a list of texts using Arctic Embeddings model.

    Args:
        request: Embedding request containing texts and configuration

    Returns:
        Embedding response containing list of embeddings

    """

    # handle 0 input case
    if len(request.texts) == 0:
        return EmbeddingResponse(embeddings=[])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_pooling_layer=False,
        safe_serialization=True,
    )

    model = model.to(device)
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
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        embeddings = model(**tokens)[0][:, 0]

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Move back to CPU for serialization
    embeddings = embeddings.cpu()

    return EmbeddingResponse(embeddings=embeddings.tolist())


@app.local_entrypoint()
async def embeddings_test():
    """
    Test the embedding function to see if it gives reasonable results.
    """
    test_inputs = [
        "How do I use embeddings for search?",
        "What is the best way to implement vector search?",
        "How to make pancakes from scratch",
    ]

    request = EmbeddingRequest(texts=test_inputs, is_query=True)

    print("Running get_embeddings locally...")

    # Now we can directly await the async function
    response = await get_embeddings.local(request)

    print(f"Generated {len(response.embeddings)} embeddings")
    print(f"Embedding dimension: {len(response.embeddings[0])}")

    emb1 = torch.tensor(response.embeddings[0])
    emb2 = torch.tensor(response.embeddings[1])
    emb3 = torch.tensor(response.embeddings[2])

    sim_12 = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    sim_13 = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0))

    print(f"Similarity between search-related texts: {sim_12.item():.4f}")
    print(f"Similarity between search and pancakes: {sim_13.item():.4f}")
