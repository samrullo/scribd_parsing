import re
import pathlib

# let's try to use nltk sentence tokenizer to tokenize our text into sentences
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


def format_number(number):
    return f"{number:,.2f}"


def read_text(text_folder: pathlib.Path):
    files = [file for file in text_folder.iterdir() if "txt" in file.suffix]
    all_text_list = [file.read_text(encoding="utf-8") for file in files]
    all_text = " ".join(all_text_list)

    # this matches two words separated with hyphen and spaces
    # it will utilize matching groups to concatenate two groups represented by two words
    repaired_text = re.sub(r"(\w+)\-\s+(\w+)", r"\1\2", all_text)

    # remove duplicate spaces
    repaired_text = re.sub(r"\s+", " ", repaired_text)

    return repaired_text


def get_tokenized_sentences(repaired_text: str):
    sentences = sent_tokenize(repaired_text)
    print(f"tokenized text into total of {format_number(len(sentences))} sentences")
    return sentences


def get_english_stop_words():
    return set(stopwords.words("english"))


def get_tokenized_words(sentence: str):
    return word_tokenize(sentence)


def filter_out_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]

def get_words_from_sentence(sentence:str):
    words = get_tokenized_words(sentence)
    return filter_out_stopwords(words,get_english_stop_words())

def get_pos_tagged_words(words):
    return nltk.pos_tag(words)

def get_nouns_from_pos_tagged_words(pos_tagged_words):
    return [word for word,tag in pos_tagged_words if tag in ('NN','NNS','NNP','NNPS')]