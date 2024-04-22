import math
from typing import Iterable, List
import json
import torch
import os
import numpy as np
from sentence_transformers.models.Transformer import Transformer
from sentence_transformers import SentenceTransformer



def merge(sparse_vectors: Iterable[dict]) -> dict:
    merged = {}
    for vec in sparse_vectors:
        # Sum attention probabilities within the same chunk
        # But compute max attention probability between chunks
        # So that repeating of the same word in the same chunk
        # will not dilute the attention probability
        aggregated_vector = {}
        for k, v in vec.items():
            aggregated_vector[k] = aggregated_vector.get(k, 0) + v

        for k, v in aggregated_vector.items():
            merged[k] = merged.get(k, 0) + v

    return merged

class SparseModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        model_path = f'cache/models--{model_name.replace("/", "--")}'

        if not os.path.exists(model_path):
            model = SentenceTransformer(model_name, cache_folder="cache")


        # Find folder which contains `sentence_bert_config.json` in model_path
        for root, dirs, files in os.walk(model_path):
            if 'sentence_bert_config.json' in files:
                model_path = root
                break
    
        sentence_bert_config = json.load(
            open(f'{model_path}/sentence_bert_config.json', 'r'))

        # sentence_bert_config['model_args'] = {
        #     "output_attention": True,
        #     **sentence_bert_config.get('model_args', {})
        # }

        self.transformer = Transformer(
            model_path,
            **sentence_bert_config
        )

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.transformer.auto_model = self.transformer.auto_model.cuda()

        self.invert_vocab = {}

        for token, idx in self.transformer.tokenizer.vocab.items():
            self.invert_vocab[idx] = token

        self.special_tokens = self.transformer.tokenizer.all_special_tokens
        self.special_tokens_ids = self.transformer.tokenizer.convert_tokens_to_ids(
            self.special_tokens)

    def encode_raw(self, sentences):
        if len(sentences) == 0:
            return [{
                'token_ids': [],
                'tokens': [],
                'weights': []
            }]

        features = self.transformer.tokenize(sentences)
        attention_mask = features['attention_mask']
        input_ids = features['input_ids']
        trans_features = {
            'input_ids': features['input_ids'],
            'attention_mask': features['attention_mask']
        }

        if self.use_cuda:
            trans_features['input_ids'] = trans_features['input_ids'].cuda()
            trans_features['attention_mask'] = trans_features['attention_mask'].cuda()

        attentions = self.transformer.auto_model(
            **trans_features, output_attentions=True).attentions[-1].cpu()

        # Shape: (batch_size, max_seq_length)
        weights = torch.mean(attentions[:, :, 0], axis=1) * attention_mask

        text_tokens = []

        for sentence_token_ids in input_ids:
            tokens = []
            for token_id in sentence_token_ids:
                tokens.append(self.invert_vocab[int(token_id)])
            text_tokens.append(tokens)
        
        result = []

        weights = weights.detach().cpu()
        attention_mask = attention_mask.bool()

        for sentence_mask, token_ids, tokens, weight in zip(attention_mask, input_ids, text_tokens, weights):
            token_ids = token_ids[sentence_mask].cpu().numpy().tolist()
            weight = weight[sentence_mask].cpu().numpy().tolist()
            
            result.append({
                'token_ids': token_ids,
                'tokens': tokens[:len(token_ids)],
                'weights': weight
            })

        return result

    def encode(self, sentences, as_tokens=False) -> Iterable[dict]:
        
        input_ids, weights = self.encode_raw(sentences)

        for i in range(len(sentences)):
            sentence_weights = {}
            for j, token_id in enumerate(input_ids[i]):
                if token_id in self.special_tokens_ids:
                    continue

                if as_tokens:
                    token_id = self.invert_vocab[int(token_id)]
                else:
                    token_id = int(token_id)

                token_weight = float(weights[i][j])

                if token_weight > 0.0:
                    sentence_weights[token_id] = sentence_weights.get(
                        token_id, 0) + token_weight

            yield sentence_weights

    def encode_documents(self, documents: List[List[str]], as_tokens=False) -> Iterable[dict]:
        batch = []
        mapping = {}
        n = 0

        for doc_id, doc in enumerate(documents):
            for sentence in doc:
                batch.append(sentence)
                mapping[n] = doc_id
                n += 1

        merged = {}
        flatten_vectors = self.encode(batch, as_tokens=as_tokens)

        for i, vec in enumerate(flatten_vectors):
            doc_id = mapping[i]
            merged[doc_id] = merge([merged.get(doc_id, {}), vec])

        for doc_id in range(len(documents)):
            yield merged.get(doc_id, {})
        


def main():
    model = SparseModel()
    documents = [
        'I am a cat.',
        'Hello, I am a dog.'
    ]

    for doc in model.encode_raw(documents):
        for key, value in doc.items():
            print(key, value)
        print()

if __name__ == '__main__':
    main()