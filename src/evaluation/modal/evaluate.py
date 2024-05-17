from dataclasses import dataclass
from itertools import product
from modal import App, Image, gpu, Mount, Secret

from src.database.database_utils import get_weaviate_client
from src.evaluation.retrieval_evaluation import execute_evaluation
from src.preprocessor.preprocessing import FileIO
from src.reranker import ReRanker

import os

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Baseline for the course
    "BAAI/bge-base-en-v1.5",  # Higher ranked model on the MTEB
]
RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Baseline for the course
    # "BAAI/bge-reranker-base",  # Higher ranked model on the MTEB
]
CHUNK_SIZES = [128, 256, 512]
LIMITS = [50, 100, 250]
ALPHAS = [0.25, 0.5, 0.75]

# GPU Configuration
gpu_config = gpu.T4()

current_working_directory = os.getcwd()
DATA_DIR = os.path.join(current_working_directory, "data")
RESULTS_DIR = os.path.join(current_working_directory, "eval_results")


def download_models():
    from sentence_transformers import SentenceTransformer

    for embedding_model in EMBEDDING_MODELS:
        SentenceTransformer(embedding_model)
    for reranker_model in EMBEDDING_MODELS:
        ReRanker(reranker_model)


app = App("uplimit-evaluation")
image = (
    Image.debian_slim()
    .pip_install_from_requirements("./requirements.txt")
    .run_function(download_models)
)


@dataclass
class ModelConfig:
    embedding_model_name: str
    reranker_model_name: int
    chunk_size: int
    limit: int
    alpha: float


def generate_configs():
    for embedding_model_name, reranker_model_name, chunk_size, limit, alpha in product(
        EMBEDDING_MODELS, RERANKER_MODELS, CHUNK_SIZES, LIMITS, ALPHAS
    ):
        yield ModelConfig(
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            chunk_size=chunk_size,
            limit=limit,
            alpha=alpha,
        )


@app.function(
    image=image,
    gpu=gpu_config,
    concurrency_limit=10,
    allow_concurrent_inputs=True,
    timeout=1200,  # in seconds, corresponds to 20 minutes
    secrets=[Secret.from_dotenv()],
    mounts=[Mount.from_local_dir(DATA_DIR, remote_path="/data")],
)
def evaluate(
    config: ModelConfig,
):
    embedding_model_name = config.embedding_model_name
    reranker_model_name = config.reranker_model_name
    chunk_size = config.chunk_size
    limit = config.limit
    alpha = config.alpha

    base_model_name = (
        embedding_model_name.split("/")[1].lower().replace("-", "_").replace(".", "")
    )
    collection_name = f"Huberman_{base_model_name}_{chunk_size}"
    client = get_weaviate_client(model_name_or_path=embedding_model_name)
    all_collections = client.show_all_collections()

    if collection_name not in all_collections:
        return {}

    print(
        f"Running test on embedding model {embedding_model_name}, reranker model {reranker_model_name}, chunk size {chunk_size}, limit {limit}, alpha {alpha}..."
    )

    reranker = ReRanker(model_name=reranker_model_name)
    data_path = f"/data/golden_datasets/golden_{chunk_size}.json"
    golden_dataset = FileIO().load_json(data_path)

    return execute_evaluation(
        dataset=golden_dataset,
        collection_name=collection_name,
        retriever=client,
        reranker=reranker,
        chunk_size=chunk_size,
        retrieve_limit=limit,
        alpha=alpha,
        dir_outpath=None,
    )


@app.local_entrypoint()
def main():
    import pandas as pd
    import time

    date = time.strftime("%Y-%m-%d-%H-%M")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for experiment_result in evaluate.map(
        generate_configs(), order_outputs=True, return_exceptions=True
    ):
        if isinstance(experiment_result, Exception):
            print(f"Encountered Exception of {experiment_result}")
            continue
        results.append(experiment_result)

        # This is to ensure that the results are not lost if the job is interrupted
        df = pd.DataFrame(results)
        df.to_csv(
            f"{RESULTS_DIR}/{date}_evaluation_results.csv",
            index=False,
        )
