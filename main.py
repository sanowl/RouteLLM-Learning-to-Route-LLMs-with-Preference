import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import json
import requests
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            item['query'],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(item['label'], dtype=torch.float)
        }

class SimilarityWeightedRanking(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.similarity = nn.CosineSimilarity(dim=1)
        self.classifier = nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        similarities = self.similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        return self.classifier(similarities.mean(dim=1).unsqueeze(1))

class MatrixFactorization(nn.Module):
    def __init__(self, num_queries, num_models, embedding_dim=64):
        super().__init__()
        self.query_embedding = nn.Embedding(num_queries, embedding_dim)
        self.model_embedding = nn.Embedding(num_models, embedding_dim)

    def forward(self, query_ids, model_ids):
        query_embed = self.query_embedding(query_ids)
        model_embed = self.model_embedding(model_ids)
        return torch.sum(query_embed * model_embed, dim=1)

class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class RouterTrainer:
    def __init__(self, model, device, learning_rate=2e-5):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, val_loader, epochs):
        best_val_accuracy = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_router_model.pth')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs.squeeze(), labels)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
                
                preds = torch.round(torch.sigmoid(outputs.squeeze()))
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

class APIModel:
    def __init__(self, model_type, api_key):
        self.model_type = model_type
        self.api_key = api_key

    def generate(self, prompt):
        if self.model_type == 'gpt4':
            return self.call_gpt4_api(prompt)
        elif self.model_type == 'claude':
            return self.call_claude_api(prompt)
        elif self.model_type == 'llama':
            return self.call_llama_api(prompt)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def call_gpt4_api(self, prompt):
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    def call_claude_api(self, prompt):
        anthropic = Anthropic(api_key=self.api_key)
        response = anthropic.completions.create(
            model="claude-3.5-20240229",
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            max_tokens_to_sample=100
        )
        return response.completion

    def call_llama_api(self, prompt):
        response = requests.post(
            "https://api.together.xyz/inference",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "togethercomputer/llama-2-70b",
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7
            }
        )
        return response.json()['output']['choices'][0]['text'].strip()

class OpenSourceModel:
    def __init__(self, model_name='mistralai/Mixtral-8x7B-v0.1'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class LLMBlender:
    def __init__(self, models):
        self.models = models
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def generate_responses(self, query):
        responses = []
        for model in self.models.values():
            responses.append(model.generate(query))
        return responses

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def calculate_coherence(self, response):
        sentences = sent_tokenize(response)
        if len(sentences) < 2:
            return 0
        embeddings = self.get_embeddings(sentences)
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            coherence_scores.append(cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0])
        return np.mean(coherence_scores)

    def rank_responses(self, query, responses):
        query_embedding = self.get_embeddings([query])[0]
        response_embeddings = self.get_embeddings(responses)
        
        scores = []
        for i, response in enumerate(responses):
            length_score = len(response) / max(len(r) for r in responses)
            relevance_score = cosine_similarity([query_embedding], [response_embeddings[i]])[0][0]
            coherence_score = self.calculate_coherence(response)
            
            combined_score = 0.3 * length_score + 0.4 * relevance_score + 0.3 * coherence_score
            scores.append(combined_score)
        
        return responses[np.argmax(scores)]

    def blend(self, query):
        responses = self.generate_responses(query)
        return self.rank_responses(query, responses)

class PromptEngineer:
    def __init__(self):
        self.templates = {
            "general": "Please answer the following question: {query}",
            "math": "Solve the following math problem step by step: {query}",
            "coding": "Write a function to solve the following programming problem: {query}",
            "analysis": "Provide a detailed analysis of the following topic: {query}"
        }

    def classify_query(self, query):
        if any(keyword in query.lower() for keyword in ["calculate", "solve", "equation"]):
            return "math"
        elif any(keyword in query.lower() for keyword in ["function", "code", "program"]):
            return "coding"
        elif any(keyword in query.lower() for keyword in ["analyze", "explain", "discuss"]):
            return "analysis"
        else:
            return "general"

    def engineer_prompt(self, query):
        query_type = self.classify_query(query)
        return self.templates[query_type].format(query=query)

class EnhancedRouteLLM:
    def __init__(self, router_models, models, tokenizer, device, threshold=0.5):
        self.router_models = router_models
        self.models = models
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.blender = LLMBlender(self.models)
        self.prompt_engineer = PromptEngineer()

    def route_query(self, query):
        engineered_prompt = self.prompt_engineer.engineer_prompt(query)
        inputs = self.tokenizer(engineered_prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        routing_scores = []
        with torch.no_grad():
            for router_model in self.router_models:
                router_output = router_model(input_ids, attention_mask)
                routing_scores.append(torch.sigmoid(router_output).item())
        
        avg_routing_score = sum(routing_scores) / len(routing_scores)
        
        if avg_routing_score > self.threshold:
            return self.blender.blend(engineered_prompt)
        else:
            return self.models['opensource'].generate(engineered_prompt)

    def chain_of_thought(self, query):
        engineered_prompt = self.prompt_engineer.engineer_prompt(query)
        steps = self.route_query(engineered_prompt).split('. ')
        detailed_steps = []
        for step in steps:
            detailed_steps.append(self.route_query(self.prompt_engineer.engineer_prompt(step)))
        return ' '.join(detailed_steps)

def evaluate_enhanced_routellm(routellm, test_data):
    total_queries = len(test_data)
    correct_predictions = 0
    blender_calls = 0
    
    for item in tqdm(test_data, desc="Evaluating EnhancedRouteLLM"):
        query = item['query']
        expected_response = item['response']
        
        routed_response = routellm.route_query(query)
        if routed_response == expected_response:
            correct_predictions += 1
            if routellm.route_query(query) != routellm.models['opensource'].generate(query):
                blender_calls += 1
    
    accuracy = correct_predictions / total_queries
    blender_usage = blender_calls / total_queries
    
    return accuracy, blender_usage

def calculate_cpt(routellm, test_data, desired_pgr):
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        routellm.threshold = threshold
        accuracy, blender_usage = evaluate_enhanced_routellm(routellm, test_data)
        if accuracy >= desired_pgr:
            return blender_usage
    return 1.0  # If desired PGR is not achievable

# ... (All previous code remains the same)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    with open('preference_data.json', 'r') as f:
        data = json.load(f)
    
    # Initialize tokenizer and router models
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    router_models = [
        SimilarityWeightedRanking().to(device),
        RoBERTaClassifier().to(device),
        MatrixFactorization(len(data), 4).to(device)  # 4 models: GPT-4, Claude, Llama, and Mixtral
    ]
    
    # Initialize API and open-source models
    gpt4_model = APIModel('gpt4', 'your-openai-api-key')
    claude_model = APIModel('claude', 'your-anthropic-api-key')
    llama_model = APIModel('llama', 'your-together-api-key')
    opensource_model = OpenSourceModel()
    
    models = {
        'gpt4': gpt4_model,
        'claude': claude_model,
        'llama': llama_model,
        'opensource': opensource_model
    }
    
    # Create dataset and data loaders
    dataset = PreferenceDataset(data, tokenizer)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # Train the router models
    for i, router_model in enumerate(router_models):
        print(f"Training Router Model {i+1}")
        trainer = RouterTrainer(router_model, device)
        trainer.train(train_loader, val_loader, epochs=5)
    
    # Initialize EnhancedRouteLLM
    enhanced_routellm = EnhancedRouteLLM(router_models, models, tokenizer, device)
    
    # Evaluate EnhancedRouteLLM
    with open('test_data.json', 'r') as f:
        test_data = json.load(f)
    
    accuracy, blender_usage = evaluate_enhanced_routellm(enhanced_routellm, test_data)
    print(f"EnhancedRouteLLM Accuracy: {accuracy:.4f}")
    print(f"Blender Usage: {blender_usage:.4f}")
    
    # Calculate CPT for different PGR levels
    for pgr in [0.5, 0.8]:
        cpt = calculate_cpt(enhanced_routellm, test_data, pgr)
        print(f"CPT for PGR {pgr}: {cpt:.4f}")
    
    # Example usage of chain-of-thought reasoning
    example_query = "Explain the process of photosynthesis and its importance in the ecosystem."
    cot_response = enhanced_routellm.chain_of_thought(example_query)
    print(f"Chain of Thought Response:\n{cot_response}")
    
    # Interactive mode for user queries
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() == 'exit':
            break
        response = enhanced_routellm.route_query(user_query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()