try:
    from preprocessing import FileIO
except ModuleNotFoundError:
    from src.preprocessor.preprocessing import FileIO
import tiktoken
from llama_index.text_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer
from src.database.properties_template import properties

import os
from tqdm import tqdm
import torch

from dotenv import load_dotenv, find_dotenv
from src.database.weaviate_interface_v4 import WeaviateIndexer, WeaviateWCS

load_dotenv(find_dotenv(), override=True)

import re


def strip_special_characters(input_string):
    # Use a regular expression to replace all non-alphanumeric characters with an empty string
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)


class CollectionMaker:
    def __init__(
        self,
        data_path: str = "../data/huberman_labs.json",
        endpoint: str = os.environ["WEAVIATE_ENDPOINT"],
        api_key: str = os.environ["WEAVIATE_API_KEY"],
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.data = FileIO.load_json(data_path)

    def _split_contents(
        self,
        corpus: list[dict],
        chunk_size=256,
        splitter: SentenceSplitter = None,
        content_field: str = "content",
        expanded_content_field: str = "expanded_content",
        window_size=1,
    ):
        if splitter is None:
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo-0125").encode,
                chunk_overlap=0,
            )
        output = []
        expanded_output = []
        for doc in tqdm(corpus):
            split_docs = splitter.split_text(doc[content_field])
            output.append(split_docs)
            episode_container = []
            for i, doc in enumerate(split_docs):
                start = max(0, i - window_size)
                end = i + window_size + 1
                expanded_content = " ".join(split_docs[start:end])
                episode_container.append(expanded_content)
            expanded_output.append(episode_container)
        return output, expanded_output

    def _encode_content_splits(
        self,
        content_splits: list[list[str]],
        model: SentenceTransformer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> list[list[tuple[str, list[float]]]]:

        text_vector_tuples = []

        ########################
        # START YOUR CODE HERE #
        ########################

        print(model)

        model.to(device)

        for content in tqdm(content_splits):
            vecs = model.encode(content).tolist()
            text_vector = [(t, v) for t, v in zip(content, vecs)]
            text_vector_tuples.append(text_vector)

        return text_vector_tuples

    def _join_metadata(
        self,
        corpus: list[dict],
        text_vector_list: list[list[tuple[str, list]]],
        expanded_content_list: list[list[str]],
        unique_id_field: str = "video_id",
        content_field: str = "content",
        expanded_content_field: str = "expanded_content",
        embedding_field: str = "content_embedding",
    ) -> list[dict]:
        """
        Combine episode metadata from original corpus with text/vectors tuples.
        Creates a new dictionary for each text/vector combination.
        """

        joined_documents = []

        ########################
        # START YOUR CODE HERE #
        ########################

        for i, doc in enumerate(corpus):
            for j, tv in enumerate(text_vector_list[i]):
                corp_dict = {
                    key: value for key, value in doc.items() if key != "content"
                }
                video_id = doc["video_id"]
                corp_dict["doc_id"] = f"{video_id}_{j}"
                corp_dict["content"] = tv[0]
                corp_dict["content_embedding"] = tv[1]
                corp_dict["expanded_content"] = expanded_content_list[i][j]
                joined_documents.append(corp_dict)

        return joined_documents

    def make_collection(
        self,
        model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 256,
    ):
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo-0125").encode,
            chunk_overlap=0,
        )
        model = SentenceTransformer(model_path)
        model_name_mod = strip_special_characters(model_path)
        corpus = self.data
        content_splits, expanded_content_splits = self._split_contents(
            corpus, chunk_size, splitter
        )
        text_vector_tuples = self._encode_content_splits(content_splits, model)
        joined_docs = self._join_metadata(
            corpus, text_vector_tuples, expanded_content_splits
        )
        collection_name = f"Huberman_{model_name_mod}_{chunk_size}"
        client = WeaviateWCS(
            endpoint=self.endpoint, api_key=self.api_key, model_name_or_path=model_path
        )
        client.create_collection(
            collection_name=collection_name,
            properties=properties,
            description="Huberman Labs: 193 full-length transcripts",
        )
        indexer = WeaviateIndexer(client)

        batch_object = indexer.batch_index_data(joined_docs, collection_name)

        client.close()
