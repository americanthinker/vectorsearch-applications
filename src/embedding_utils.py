from llama_index.text_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer
import time
from torch import cuda
from tqdm import tqdm

from src.preprocessor.preprocessing import FileIO


def split_contents(
    corpus: list[dict], text_splitter: SentenceSplitter, content_field: str = "content"
) -> list[list[str]]:
    """
    Given a corpus of "documents" with text content, this function splits the
    content field into chunks sizes as specified by the text_splitter.

    Example
    -------
    corpus = [
            {'title': 'This is a cool show', 'content': 'There is so much good content on this show. \
              This would normally be a really long block of content. ... But for this example it will not be.'},
            {'title': 'Another Great Show', 'content': 'The content here is really good as well.  If you are \
              reading this you have too much time on your hands. ... More content, blah, blah.'}
           ]

    output = split_contents(data, text_splitter, content_field="content")

    output >>> [['There is so much good content on this show.', 'This would normally be a really long block of content.', \
                 'But for this example it will not be'],
                ['The content here is really good as well.', 'If you are reading this you have too much time on your hands.', \
                 'More content, blah, blah.']
                ]
    """

    chunks = []
    for document in corpus:
        content = document.get(content_field, None)
        if content is None:
            raise ValueError(f"No content field found in document: {document}")
        chunks.append(text_splitter.split_text(content))
    return chunks


def encode_content_splits(
    content_splits: list[list[str]],
    model: SentenceTransformer,
    device: str = "cuda:0" if cuda.is_available() else "cpu",
) -> list[list[tuple[str, list[float]]]]:
    """
    Encode content splits as vector embeddings from a vectors of content splits
    where each vectors of splits is a single podcast episode.

    Example
    -------
    content_splits =  [['There is so much good content on this show.', 'This would normally be a really long block of content.'],
                       ['The content here is really good as well.', 'More content, blah, blah.']
                      ]

    output = encode_content_splits(content_splits, model)

    output >>> [
          EPISODE 1 -> [('There is so much good content on this show.',[ 1.78036056e-02, -1.93265956e-02,  3.61164124e-03, -5.89650944e-02,
                                                                         1.91510320e-02,  1.60808843e-02,  1.13610983e-01,  3.59948091e-02,
                                                                        -1.73066761e-02, -3.30348089e-02, -1.00898169e-01,  2.34847311e-02]
                                                                        )
                         tuple(text, vectors), tuple(text, vectors), tuple(text, vectors)....],
          EPISODE 2 ->  [tuple(text, vectors), tuple(text, vectors), tuple(text, vectors)....],
          EPISODE n ... [tuple(text, vectors), tuple(text, vectors), tuple(text, vectors)....]
    """

    text_vector_tuples = []
    for splits in tqdm(content_splits):
        episode_vectors = model.encode(
            splits, convert_to_tensor=False, convert_to_numpy=True, device=device
        )
        text_vector_tuples.append(list(zip(splits, episode_vectors.tolist())))
    return text_vector_tuples


def join_metadata(
    corpus: list[dict],
    text_vector_list: list[list[tuple[str, list]]],
    unique_id_field: str = "video_id",
    content_field: str = "content",
    embedding_field: str = "content_embedding",
) -> list[dict]:
    """
    Combine episode metadata from original corpus with text/vectors tuples.
    Creates a new dictionary for each text/vector combination.
    """

    joined_documents = []
    for i, document in enumerate(corpus):
        unique_id = document.get(unique_id_field, None)
        if unique_id is None:
            raise ValueError(f"No unique_id field found in document: {document}")

        text_vector_tuples = text_vector_list[i]
        for j, (text, vector) in enumerate(text_vector_tuples):
            new_document = document.copy()
            new_document["doc_id"] = f"{unique_id}_{j}"
            new_document[content_field] = text
            new_document[embedding_field] = vector
            joined_documents.append(new_document)
    return joined_documents


def create_dataset(
    corpus: list[dict],
    embedding_model: SentenceTransformer,
    text_splitter: SentenceSplitter,
    save_to_disk: bool,
    file_outpath: str = None,
    unique_id_field: str = "video_id",
    content_field: str = "content",
    embedding_field: str = "content_embedding",
    device: str = "cuda:0" if cuda.is_available() else "cpu",
) -> list[dict]:
    """
    Given a raw corpus of data, this function creates a new dataset where each dataset
    doc contains episode metadata and it's associated text chunk and vector representation.
    Output is directly saved to disk.
    """
    if save_to_disk and not file_outpath:
        raise ValueError(
            "Saving to disk is enabled but file_outpath was left as a None value.\n\
            Enter a valid file_outpath or mark save_to_disk as False"
        )

    io = FileIO()

    chunk_size = text_splitter.chunk_size
    print(f"Creating dataset using chunk_size: {chunk_size}")
    start = time.perf_counter()

    content_splits = split_contents(
        corpus=corpus, text_splitter=text_splitter, content_field=content_field
    )
    text_vector_tuples = encode_content_splits(
        content_splits=content_splits, model=embedding_model, device=device
    )
    joined_docs = join_metadata(
        corpus=corpus,
        text_vector_list=text_vector_tuples,
        unique_id_field=unique_id_field,
        content_field=content_field,
        embedding_field=embedding_field,
    )

    if save_to_disk:
        io.save_as_parquet(file_path=file_outpath, data=joined_docs, overwrite=False)
    end = time.perf_counter() - start
    print(
        f"Total Time to process dataset of chunk_size ({chunk_size}): {round(end/60, 2)} minutes"
    )
    return joined_docs
