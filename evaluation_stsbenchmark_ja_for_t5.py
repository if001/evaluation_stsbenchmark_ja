"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset

Usage:
python evaluation_stsbenchmark_ja.py
OR
python evaluation_stsbenchmark_ja.py model_name
"""
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from transformers import T5Tokenizer, T5Model
from transformers import MLukeTokenizer, LukeModel


from datasets import load_dataset
import logging
import sys
import torch
import os


class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="max_length", max_length=128,
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)
    
class SentenceT5:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, is_fast=False)
        self.model = T5Model.from_pretrained(model_name_or_path).encoder
        # self.model = T5Model.from_pretrained(model_name_or_path, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto").encoder
        
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8, show_progress_bar=False, convert_to_numpy=True):
        all_embeddings = []
        batch_size = 1
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="max_length", max_length=128,
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        result = torch.stack(all_embeddings)
        # if convert_to_numpy:
        #     result = result.to('cpu').detach().numpy().copy()
        return result
    

#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# model_name ='sonoisa/sentence-bert-base-ja-mean-tokens-v2'
model_name = "sonoisa/sentence-t5-base-ja-mean-tokens"
t5_model = SentenceT5(model_name, 'cpu')

ds = load_dataset('stsb_multi_mt_ja', 'ja', split='test')

sentences1 = ds['sentence1']
sentences2 = ds['sentence2']
scores = [x/5.0 for x in ds['similarity_score']]

print(sentences1[:3], sentences2[:3])

results = []

print(model_name)
# model = SentenceTransformer(model_name, device="cuda")
# model = SentenceTransformer(modules=[t5_model], device="cuda")
evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, main_similarity=SimilarityFunction.COSINE, name='sts-test')
# spearman_cos = model.evaluate(evaluator)
spearman_cos = evaluator(t5_model)
results.append('| {:s} | {:.1f} |'.format(model_name, spearman_cos * 100))

print('| model | spearman_cos |')
print('|-------|-------------:|')
for result in results:
    print(result)
