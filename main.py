import torch, torch.nn as nn, torch.optim as optim
from transformers import AutoModel, AutoTokenizer, pipeline
from typing import List, Tuple, Dict, Callable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
from ray import tune
import pandas as pd 

@dataclass
class LLMConfig: name: str; model_name: str; cost_per_token: float; quality: float

class LLM:
    def __init__(self, config: LLMConfig):
        self.config, self.model, self.tokenizer = config, AutoModel.from_pretrained(config.model_name), AutoTokenizer.from_pretrained(config.model_name)
        self.generator = pipeline('text-generation', model=config.model_name, tokenizer=config.model_name)
    def generate(self, query: str, max_length: int = 150) -> str: return self.generator(query, max_length=max_length, num_return_sequences=1)[0]['generated_text']
    def embed(self, query: str) -> torch.Tensor:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad(): return self.model(**inputs).last_hidden_state.mean(dim=1).squeeze()

class AdvancedRouter(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_models: int, num_layers: int = 2):
        super().__init__()
        self.lstm, self.attention = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True), nn.MultiheadAttention(hidden_size * 2, 8)
        self.fc1, self.fc2, self.dropout = nn.Linear(hidden_size * 2, hidden_size), nn.Linear(hidden_size, num_models), nn.Dropout(0.2)
    def forward(self, x, attention_mask):
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=~attention_mask.bool())
        return self.fc2(self.dropout(torch.relu(self.fc1(attn_output.mean(dim=0)))))

class RouteLLMDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]]): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class RouteLLM:
    def __init__(self, models: List[LLM], device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.models, self.device = models, device
        self.router = AdvancedRouter(768, 256, len(models)).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.optimizer, self.scheduler = optim.AdamW(self.router.parameters(), lr=2e-5), optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2)
        self.loss_fn = nn.CrossEntropyLoss()

    def route(self, query: str) -> LLM:
        self.router.eval()
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            return self.models[self.router(**inputs).argmax().item()]

    def train(self, data: List[Tuple[str, int]], epochs: int = 10, batch_size: int = 32, val_split: float = 0.1):
        dataset = RouteLLMDataset(data)
        train_size = int((1 - val_split) * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.collate_fn)
        self.router.train()
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = 0.0
            for batch in tqdm(train_dataloader, desc="Training"):
                inputs, labels, attention_mask = batch
                self.optimizer.zero_grad()
                outputs = self.router(inputs, attention_mask)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            self.router.eval()
            val_loss = sum(self.loss_fn(self.router(inputs, attention_mask), labels).item() for inputs, labels, attention_mask in val_dataloader) / len(val_dataloader)
            self.router.train()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss: best_val_loss = val_loss; torch.save(self.router.state_dict(), 'best_router.pt')

    def collate_fn(self, batch):
        queries, labels = zip(*batch)
        encodings = self.tokenizer(list(queries), padding=True, truncation=True, max_length=512, return_tensors='pt')
        return encodings['input_ids'], torch.tensor(labels), encodings['attention_mask']

    def evaluate(self, data: List[Tuple[str, str]]) -> Dict[str, float]:
        self.router.eval()
        predictions, truths, qualities, costs = [], [], [], []
        for query, expected in tqdm(data, desc="Evaluating"):
            model = self.route(query)
            prediction = self.models.index(model)
            truth = next(i for i, m in enumerate(self.models) if m.generate(query) == expected)
            qualities.append(model.config.quality)
            costs.append(model.config.cost_per_token)
            predictions.append(prediction)
            truths.append(truth)
        precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, average='weighted')
        return {"accuracy": accuracy_score(truths, predictions), "precision": precision, "recall": recall, "f1_score": f1, "avg_quality": np.mean(qualities), "avg_cost": np.mean(costs)}

    def calculate_metrics(self, data: List[Tuple[str, str]], metric_fns: Dict[str, Callable]) -> Dict[str, float]:
        results = self.evaluate(data)
        return {**results, **{name: fn(results) for name, fn in metric_fns.items()}}

    def visualize_performance(self, results: Dict[str, float]):
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(results.keys()), y=list(results.values()))
        plt.title("RouteLLM Performance Metrics")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('routellm_performance.png')
        plt.close()

    def analyze_routing_decisions(self, data: List[Tuple[str, str]]) -> pd.DataFrame:
        decisions = [{"query": query, "routed_to": model.config.name, "model_quality": model.config.quality, "model_cost": model.config.cost_per_token} for query, _ in data if (model := self.route(query))]
        return pd.DataFrame(decisions).to_csv('routing_decisions.csv', index=False) or pd.DataFrame(decisions)

# Example usage
configs = [LLMConfig("BERT", "bert-base-uncased", 0.001, 0.7), LLMConfig("GPT2", "gpt2", 0.005, 0.8), LLMConfig("RoBERTa", "roberta-base", 0.008, 0.85), LLMConfig("BART", "facebook/bart-base", 0.01, 0.9)]
models = [LLM(config) for config in configs]
route_llm = RouteLLM(models)

train_data = [("What is the capital of France?", 0), ("Explain quantum entanglement", 3), ("Translate 'Hello' to French", 2), ("What is the plot of Hamlet?", 1)]
route_llm.train(train_data, epochs=20)

eval_data = [("What is 2+2?", "BERT response: The result of 2+2 is 4."), ("Explain the theory of relativity", "BART response: The theory of relativity, proposed by Albert Einstein, describes how..."), ("Write a poem about spring", "GPT2 response: Blossoms bloom in gentle breeze,\nSunshine warms the waking trees...")]
metric_fns = {"cost_efficiency": lambda r: r['avg_quality'] / r['avg_cost'], "performance_index": lambda r: (r['accuracy'] + r['f1_score']) * r['avg_quality']}

results = route_llm.calculate_metrics(eval_data, metric_fns)
print("Evaluation results:", results)
route_llm.visualize_performance(results)
routing_analysis = route_llm.analyze_routing_decisions(eval_data)
print("Routing analysis:", routing_analysis.head())

def objective(config):
    route_llm = RouteLLM(models)
    route_llm.router.fc1, route_llm.router.fc2 = nn.Linear(512, config['hidden_size']), nn.Linear(config['hidden_size'], len(models))
    route_llm.train(train_data, epochs=5, batch_size=config['batch_size'])
    results = route_llm.evaluate(eval_data)
    return {"score": results['f1_score'], "cost": -results['avg_cost']}

analysis = tune.run(objective, config={"hidden_size": tune.choice([64, 128, 256]), "batch_size": tune.choice([16, 32, 64])}, num_samples=10, resources_per_trial={"cpu": 2, "gpu": 0.5})
print("Best hyperparameters:", analysis.best_config)
