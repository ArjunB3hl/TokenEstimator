import json
import pandas as pd
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import math

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_json_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)



def get_embedding(text):
    """Get embedding for a text using OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 3072  # Dimension of text-embedding-3-large

def create_dataset():
    """Create a combined dataset from all JSON files"""
    # List of model JSON files to process
    model_files = [
        'gpt-3.5-turbo-data.json',
        'gpt-4o-mini-data.json',
        'o1-mini-data.json',
        'o3-mini-data.json',
        'claude-data.json'
    ]
    
    # Combine data from all model files
    all_data = []
    
    for file in model_files:
        if os.path.exists(file):
            try:
                model_name = file.replace('-data.json', '')
                data = load_json_data(file)
                
                # Process each entry in the JSON data
                for entry in data:
                    # Assuming the JSON structure has 'prompt' and 'completion_tokens'
                    if 'prompt' in entry and 'completion_tokens' in entry:
                        prompt = entry['prompt']
                        completion_tokens = entry['completion_tokens']
                        
                        all_data.append({
                            'prompt': prompt,
                            'completion_tokens': completion_tokens,
                            'model': model_name
                        })
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Generate embeddings for each prompt
    print("Generating embeddings...")
    embeddings = []
    for prompt in df['prompt']:
        embedding = get_embedding(prompt)
        embeddings.append(embedding)
    
    # Save embeddings to a file
    with open('prompt_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save the DataFrame
    df.to_csv('combined_model_data.csv', index=False)
    
    return df, embeddings

# Gated Linear Unit (GLU) implementation
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

# Neural Network for token prediction with more complexity
class TokenPredictionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_models):
        super(TokenPredictionModel, self).__init__()
        
        # Embedding processing layers with GLU
        self.embedding_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            GLU(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim)
        )
        
        # Model embedding with more dimensions
        self.model_embedding = nn.Embedding(num_models, 64)
        
        # Deeper combined layers with GLU
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim * 2),
            GLU(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            GLU(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.25),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2 * 2),
            GLU(hidden_dim // 2 * 2, hidden_dim // 2),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4 * 2),
            GLU(hidden_dim // 4 * 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, embedding, model_idx):
        # Process text embedding
        embedding_features = self.embedding_layer(embedding)
        
        # Get model embedding
        model_features = self.model_embedding(model_idx)
        
        # Combine features
        combined = torch.cat([embedding_features, model_features], dim=1)
        
        # Final prediction
        output = self.combined_layer(combined)
        return output.squeeze()

# Custom dataset for PyTorch
class TokenPredictionDataset(Dataset):
    def __init__(self, embeddings, model_indices, targets):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.model_indices = torch.tensor(model_indices, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'model_idx': self.model_indices[idx],
            'target': self.targets[idx]
        }

def train_model(df, embeddings):
    """Train a neural network model to predict completion tokens"""
    # Encode categorical features
    model_encoder = LabelEncoder()
    
    model_indices = model_encoder.fit_transform(df['model'])
    targets = df['completion_tokens'].values
    
    # Save encoder for later use
    with open('model_encoder.pkl', 'wb') as f:
        pickle.dump(model_encoder, f)
    
    # Split data into train (50%), validation (20%), and test (30%)
    # First split into train and temp (test+validation)
    X_train_indices, X_temp_indices, y_train, y_temp = train_test_split(
        np.arange(len(embeddings)), targets, test_size=0.5, random_state=42
    )
    
    # Then split temp into validation and test
    test_size_adjusted = 0.3 / 0.5  # 30% of total = 60% of temp
    X_val_indices, X_test_indices, y_val, y_test = train_test_split(
        X_temp_indices, y_temp, test_size=test_size_adjusted, random_state=42
    )
    
    print(f"Data split: Train={len(X_train_indices)} (50%), Validation={len(X_val_indices)} (20%), Test={len(X_test_indices)} (30%)")
    
    # Create datasets
    train_dataset = TokenPredictionDataset(
        [embeddings[i] for i in X_train_indices],
        model_indices[X_train_indices],
        y_train
    )
    
    val_dataset = TokenPredictionDataset(
        [embeddings[i] for i in X_val_indices],
        model_indices[X_val_indices],
        y_val
    )
    
    test_dataset = TokenPredictionDataset(
        [embeddings[i] for i in X_test_indices],
        model_indices[X_test_indices],
        y_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize model with larger hidden dimensions
    embedding_dim = len(embeddings[0])  # Dimension of OpenAI embeddings
    hidden_dim = 512  # Increased from 256
    num_models = len(model_encoder.classes_)
    
    model = TokenPredictionModel(embedding_dim, hidden_dim, num_models)
    
    # Use MSE loss and optimizer with weight decay for regularization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Training loop with 100 epochs
    num_epochs = 100
    print("Training model...")
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                batch['embedding'],
                batch['model_idx']
            )
            
            # Calculate loss
            loss = criterion(outputs, batch['target'])
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch['embedding'],
                    batch['model_idx']
                )
                loss = criterion(outputs, batch['target'])
                val_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'token_prediction_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch['embedding'],
                batch['model_idx']
            )
            loss = criterion(outputs, batch['target'])
            test_loss += loss.item()
    
    print(f"Final Test Loss: {test_loss/len(test_loader):.4f}")
    
    # Save model architecture info
    model_info = {
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_models': num_models
    }
    
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    return model, model_encoder

if __name__ == "__main__":
    print("Creating dataset...")
    df, embeddings = create_dataset()
    
    print("Training model...")
    model, model_encoder = train_model(df, embeddings)
    
    print("Done! Model saved as 'token_prediction_model.pth'")
