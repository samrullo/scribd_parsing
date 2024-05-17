import sys

sys.path.append(".")

import pathlib
import chromadb
from chromadb.utils import embedding_functions
from chroma_utils import build_chroma_collection
from text_utils import (
    read_text,
    get_tokenized_sentences,
    get_nouns_from_pos_tagged_words,
    get_pos_tagged_words,
    get_words_from_sentence,
)

text_folder = pathlib.Path.cwd() / "data" / "steven_king_books" / "doctor_sleep"
text = read_text(text_folder)
sentences = get_tokenized_sentences(text)
metadatas = [
    {
        "keyword": ",".join(get_nouns_from_pos_tagged_words(
            get_pos_tagged_words(get_words_from_sentence(sentence))
        ))
    }
    for sentence in sentences
]


chroma_path = pathlib.Path.cwd() / "chromadb"
embedding_func_name = "multi-qa-MiniLM-L6-cos-v1"
collection_name = "doctor_sleep_book"

build_chroma_collection(
    str(chroma_path),
    collection_name,
    embedding_func_name,
    sentences,
    metadatas
)
