import nltk
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords

from typing import Dict, List

from inference import SparseModel

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


sparse_model = SparseModel()


def group_bpe_by_tokens(tokens: list[str], bpe_tokens: list[str]) -> List[List[int]]:
    result = []

    
    tokens_iter = iter(tokens)
    bpe_tokens_iter = enumerate(iter(bpe_tokens))

    current_token = next(tokens_iter)
    idx, current_bpe_token = next(bpe_tokens_iter, (None, None))

    while current_token is not None and current_bpe_token is not None:

        if current_bpe_token in sparse_model.special_tokens:
            idx, current_bpe_token = next(bpe_tokens_iter, (None, None))
            continue

        if current_bpe_token == current_token:
            result.append([idx])
            idx, current_bpe_token = next(bpe_tokens_iter, (None, None))
            current_token = next(tokens_iter, None)
            continue

        if current_token.startswith(current_bpe_token):
            current_mapping = [idx]
            idx, current_bpe_token = next(bpe_tokens_iter, None)
            while current_bpe_token.startswith("##"):
                current_mapping.append(idx)
                idx, current_bpe_token = next(bpe_tokens_iter, (None, None))
            result.append(current_mapping)
            current_token = next(tokens_iter, None)
            continue
        
        idx, current_bpe_token = next(bpe_tokens_iter, (None, None))
    
    return result


def weight_snowball_with_sparse(texts: List[str]) -> Dict[str, float]:
    batch_tokens = [word_tokenize(text.lower()) for text in texts]

    batch_token_ids, batch_weights = sparse_model.encode_raw(texts)

    for tokens, token_ids, weights in zip(batch_tokens, batch_token_ids, batch_weights):
        
        print(tokens)

        bpe_tokens = []
        for token_id, weight in zip(token_ids, weights):
            token_text = sparse_model.invert_vocab[int(token_id)]

            bpe_tokens.append(token_text)

            print(f"{token_text} ({token_id}) - {weight}")

        token_groups = group_bpe_by_tokens(tokens, bpe_tokens)

        print(token_groups)

        assert len(token_groups) == len(tokens)

        print("------")

    return {}



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
    weight_snowball_with_sparse(texts)


if __name__ == "__main__":
    main()
