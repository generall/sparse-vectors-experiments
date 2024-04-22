import time
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

from typing import Any, Dict, Iterable, List, Tuple
import string

from inference import merge


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


language = "english"

stemmer = SnowballStemmer(language)

tokenizer = WordPunctTokenizer()

stop_words = set(stopwords.words(language))
punctuation = set(string.punctuation)

special_tokens = set(["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"])


def reconstruct_bpe(bpe_tokens: Iterable[Tuple[int, str]]) -> List[Tuple[str, List[int]]]:
    
    result = []
    acc = ""
    acc_idx = []
    
    for idx, token in bpe_tokens:
        if token in special_tokens:
            continue

        if token.startswith("##"):
            acc += token[2:]
            acc_idx.append(idx)
        else:
            if acc:
                result.append((acc, acc_idx))
                acc = ""
                acc_idx = []
            acc = token
            acc_idx.append(idx)

    if acc:
        result.append((acc, acc_idx))

    return result


def snowball_tokenize(text: str) -> List[str]:
    return tokenizer.tokenize(text.lower())


def filter_list_tokens(tokens: List[str]) -> List[str]:
    result = []
    for token in tokens:
        if token in stop_words or token in punctuation:
            continue
        result.append(token)
    return result


def filter_pair_tokens(tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    result = []
    for token, value in tokens:
        if token in stop_words or token in punctuation:
            continue
        result.append((token, value))
    return result


def stem_list_tokens(tokens: List[str]) -> List[str]:
    result = []
    for token in tokens:
        processed_token = stemmer.stem(token)
        result.append(processed_token)
    return result

def stem_pair_tokens(tokens: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    result = []
    for token, value in tokens:
        processed_token = stemmer.stem(token)
        result.append((processed_token, value))
    return result


def aggregate_weights(tokens: List[Tuple[str, List[int]]], weights: List[float]) -> List[Tuple[str, float]]:
    result = []
    for token, idxs in tokens:
        sum_weight = sum(weights[idx] for idx in idxs)
        result.append((token, sum_weight))
    return result



AVG_DOC_SIZE = 100


k = 1.2
b = 0.75


def calc_tf(tf, doc_size):
    return (k + 1) * tf / (k * (1 - b + b * doc_size / AVG_DOC_SIZE) + tf)


def main():
    texts = [
        'How to pay with cash when car shopping?',
        'What credit card information are offline US merchants allowed to collect for purposes other than the transaction?',
        'How come we can find stocks with a Price-to-Book ratio less than 1?',
        'I have around 60K $. Thinking about investing in Oil, how to proceed?',
        'Can the risk of investing in an asset be different for different investors?',
        'Selling high, pay capital gains, re-purchase later',
        'How do disputed debts work on credit reports?',
        'What are some good books for learning stocks, bonds, derivatives e.t.c for beginner with a math background?',
        "Tax on Stocks or ETF's",
        'Something looks off about Mitsubishi financial data',
        'Rollover into bond fund to do dollar cost averaging [duplicate]',
        "HSBC Hong Kong's “Deposit Plus” Product: What is it, and what strategies to employ?",
        'What does “profits to the shareholders jumped to 15 cents a share” mean?',
        'What is the average cost of a portfolio on a trading site?',
        "Why government bonds fluctuate so much, even though interest rates don't change that often?",
        'At what age should I start or stop saving money?',
        'Why I cannot find a “Pure Cash” option in 401k investments?',
        'What should I do with my $25k to invest as a 20 years old?',
        'Buy on dip when earnings fail?',
        'What are the consequences of IRS “reclassification” on both employer and employee?',
        "Are there any consequences for investing in Vanguard's Admiral Shares funds instead of ETF's in a Roth IRA?",
        'What do these options trading terms mean?',
        'Is there a way to open a U.S. bank account for my LLC remotely?',
        'How to stress test an investment plan?',
        'what is the best way to do a freelancing job over the summer for a student',
        'Does financing a portfolio on margin affect the variance of a portfolio?',
        'Intentions of Deductible Amount for Small Business',
        'How do I track investment performance in Quicken across rollovers?',
        'Paid part of my state refund back last year; now must declare the initial amount as income?',
    ]


if __name__ == "__main__":
    main()
