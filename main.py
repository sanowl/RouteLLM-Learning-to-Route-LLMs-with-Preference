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
    def forward(self, input_ids, attention_mask, model_indices):
        query_emb = self.bert(input_ids, attention_mask)[1]
        query_proj = self.query_projection(query_emb)
        model_emb = self.model_embeddings(model_indices)
        return torch.sum(query_proj * model_emb, dim=1)
class CausalLLMRouter(nn.Module):
    def __init__(self):
        super(CausalLLMRouter, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, -1, :]  # Return logits for the last token

class RouteLLM:
    def __init__(self, strong_model: LLM, weak_model: LLM, router_type: str = 'bert'):
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.router_type = router_type
        self.router = self.initialize_router()
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased') if router_type != 'causal_llm' else GPT2Tokenizer.from_pretrained('gpt2')
        self.optimizer = optim.Adam(self.router.parameters(), lr=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss() if router_type != 'causal_llm' else nn.CrossEntropyLoss()
    def initialize_router(self):
        if self.router_type == 'bert':
            return BERTRouter()
        elif self.router_type == 'matrix_factorization':
            return MatrixFactorizationRouter(num_models=2, embedding_dim=128)
        elif self.router_type == 'causal_llm':
            return CausalLLMRouter()
        else:
            raise ValueError("Unsupported router type")

    def train(self, preference_data: List[Tuple[str, int]]):
        self.router.train()
        for epoch in range(5):
            total_loss = 0
            for query, label in preference_data:
                inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
                if self.router_type == 'matrix_factorization':
                    model_indices = torch.tensor([0, 1], dtype=torch.long)
                    outputs = self.router(inputs['input_ids'], inputs['attention_mask'], model_indices)
                    loss = self.loss_fn(outputs, torch.tensor([label], dtype=torch.float))
                elif self.router_type == 'causal_llm':
                    outputs = self.router(inputs['input_ids'], inputs['attention_mask'])
                    loss = self.loss_fn(outputs, torch.tensor([label], dtype=torch.long))
                else:
                    outputs = self.router(inputs['input_ids'], inputs['attention_mask'])
                    loss = self.loss_fn(outputs, torch.tensor([[label]], dtype=torch.float))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(preference_data)}")

    def route(self, query: str) -> LLM:
        self.router.eval()
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
            if self.router_type == 'matrix_factorization':
                model_indices = torch.tensor([0, 1], dtype=torch.long)
                output = self.router(inputs['input_ids'], inputs['attention_mask'], model_indices)
                prediction = torch.argmax(output).item()
            elif self.router_type == 'causal_llm':
                output = self.router(inputs['input_ids'], inputs['attention_mask'])
                prediction = torch.argmax(output).item()
            else:
                output = self.router(inputs['input_ids'], inputs['attention_mask'])
                prediction = output.item() > 0.5
            return self.strong_model if prediction else self.weak_model

    def evaluate(self, benchmark_data: List[Tuple[str, str]]) -> Dict[str, float]:
        correct = 0
        total = 0
        strong_model_usage = 0
        total_quality = 0

        for query, expected in benchmark_data:
            model = self.route(query)
            response = model.generate(query)
            if response == expected:
                correct += 1
            total += 1
            if model == self.strong_model:
                strong_model_usage += 1
            total_quality += model.quality

        accuracy = correct / total
        strong_model_percentage = strong_model_usage / total
        avg_quality = total_quality / total

        return {
            "accuracy": accuracy,
            "strong_model_usage": strong_model_percentage,
            "average_quality": avg_quality
        }

    def calculate_cost_savings(self, benchmark_data: List[Tuple[str, str]]) -> float:
        baseline_cost = len(benchmark_data) * self.strong_model.cost_per_token

        routed_cost = 0
        for query, _ in benchmark_data:
            model = self.route(query)
            routed_cost += model.cost_per_token

        savings = (baseline_cost - routed_cost) / baseline_cost
        return savings

    def calculate_cpt(self, benchmark_data: List[Tuple[str, str]], target_pgr: float) -> float:
        sorted_data = sorted(benchmark_data, key=lambda x: self.route(x[0]).quality, reverse=True)
        weak_performance = sum(self.weak_model.quality for _ in sorted_data) / len(sorted_data)
        strong_performance = sum(self.strong_model.quality for _ in sorted_data) / len(sorted_data)
        target_performance = weak_performance + target_pgr * (strong_performance - weak_performance)

        cumulative_performance = 0
        for i, (query, _) in enumerate(sorted_data):
            model = self.route(query)
            cumulative_performance += model.quality
            if cumulative_performance / (i + 1) >= target_performance:
                return (i + 1) / len(sorted_data)
        return 1.0

    def calculate_apgr(self, benchmark_data: List[Tuple[str, str]], num_points: int = 10) -> float:
        pgr_values = []
        for i in range(num_points):
            threshold = i / (num_points - 1)
            pgr = self.calculate_pgr(benchmark_data, threshold)
            pgr_values.append(pgr)
        return sum(pgr_values) / num_points

    def calculate_pgr(self, benchmark_data: List[Tuple[str, str]], threshold: float) -> float:
        routed_performance = 0
        weak_performance = 0
        strong_performance = 0
        for query, _ in benchmark_data:
            model = self.route(query)
            if model == self.strong_model and random.random() > threshold:
                model = self.weak_model
            routed_performance += model.quality
            weak_performance += self.weak_model.quality
            strong_performance += self.strong_model.quality

        routed_performance /= len(benchmark_data)
        weak_performance /= len(benchmark_data)
        strong_performance /= len(benchmark_data)

        return (routed_performance - weak_performance) / (strong_performance - weak_performance)


def process_preference_data(raw_data: List[Dict]) -> List[Tuple[str, int]]:
    processed_data = []
    for item in raw_data:
        query = item['query']
        label = 1 if item['preferred_model'] == 'strong' else 0
        processed_data.append((query, label))
    return processed_data


def augment_data(preference_data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    augmented_data = preference_data.copy()
    for query, label in preference_data:
        noisy_query = query + " " + "".join([chr(ord(c) + 1) for c in query[:5]])
        augmented_data.append((noisy_query, label))
    return augmented_data


# Example usage:
strong_model = LLM("GPT-4", cost_per_token=0.01, quality=0.9)
weak_model = LLM("GPT-3.5", cost_per_token=0.001, quality=0.7)

route_llm = RouteLLM(strong_model, weak_model, router_type='bert')

# Simulated preference data
raw_preference_data = [
    {"query": "What is the capital of France?", "preferred_model": "weak"},
    {"query": "Explain quantum entanglement", "preferred_model": "strong"},
]

preference_data = process_preference_data(raw_preference_data)
augmented_data = augment_data(preference_data)

route_llm.train(augmented_data)


benchmark_data = [
    ("What is 2+2?", "GPT-3.5 response to: What is 2+2?"),
    ("Explain the theory of relativity", "GPT-4 response to: Explain the theory of relativity"),
]

eval_results = route_llm.evaluate(benchmark_data)
print("Evaluation results:", eval_results)

cost_savings = route_llm.calculate_cost_savings(benchmark_data)
print(f"Cost savings: {cost_savings:.2%}")

cpt_50 = route_llm.calculate_cpt(benchmark_data, 0.5)
print(f"CPT(50%): {cpt_50:.2%}")

apgr = route_llm.calculate_apgr(benchmark_data)
print(f"APGR: {apgr:.2%}")
