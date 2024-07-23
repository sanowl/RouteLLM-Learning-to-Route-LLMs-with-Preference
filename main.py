import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score
import random

class LLM:
    def __init__(self, name: str, cost_per_token: float, quality: float):
        self.name = name
        self.cost_per_token = cost_per_token
        self.quality = quality

    def generate(self, query: str) -> str:
        return f"{self.name} response to: {query}"

class BERTRouter(nn.Module):
    def __init__(self):
        super(BERTRouter, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

class MatrixFactorizationRouter(nn.Module):
    def __init__(self, num_models: int, embedding_dim: int):
        super(MatrixFactorizationRouter, self).__init__()
        self.model_embeddings = nn.Embedding(num_models, embedding_dim)
        self.query_projection = nn.Linear(768, embedding_dim) 
        self.bert = BertModel.from_pretrained('bert-base-uncased')
