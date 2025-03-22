from flask import Flask, request, jsonify
import torch
import pickle
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import torch.nn as nn

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define GLU class here to avoid import issues
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

# Define TokenPredictionModel class here to avoid import issues
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

# Load the trained model and necessary files
def load_model():
    # Load model architecture info
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    # Initialize model with saved architecture
    model = TokenPredictionModel(
        model_info['embedding_dim'],
        model_info['hidden_dim'],
        model_info['num_models']
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('token_prediction_model.pth'))
    
    # Load encoder
    with open('model_encoder.pkl', 'rb') as f:
        model_encoder = pickle.load(f)
    
    return model, model_encoder

# Get embedding for a text
def get_embedding(text):
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

# Function to get prediction with confidence intervals using Monte Carlo Dropout
def predict_with_confidence(model, embedding_tensor, model_idx_tensor, num_samples=30):
    # Set model to training mode to enable dropout
    model.train()
    
    # Run multiple forward passes
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            pred = model(embedding_tensor, model_idx_tensor)
            predictions.append(pred.item())
    
    # Calculate statistics
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions)
    std_dev = np.std(predictions)
    
    # Calculate confidence intervals (95% confidence)
    lower_bound = mean_prediction - 1.96 * std_dev
    upper_bound = mean_prediction + 1.96 * std_dev
    
    # Round to nearest integers
    mean_prediction_rounded = round(mean_prediction)
    lower_bound_rounded = max(0, round(lower_bound))  # Ensure non-negative
    upper_bound_rounded = round(upper_bound)
    
    return {
        'predicted_tokens': mean_prediction_rounded,
        'confidence_interval': {
            'lower': lower_bound_rounded,
            'upper': upper_bound_rounded,
            'confidence': '95%'
        },
        'raw': {
            'mean': mean_prediction,
            'std_dev': std_dev
        }
    }

# Global variables for model and encoders
model = None
model_encoder = None

# Initialize model and encoders
def initialize():
    global model, model_encoder
    model, model_encoder = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Get request data
    data = request.json
    
    if not data or 'prompt' not in data or 'model' not in data:
        return jsonify({'error': 'Missing required fields: prompt, model'}), 400
    
    prompt = data['prompt']
    model_name = data['model']
    
    # Check if model is loaded
    global model, model_encoder
    if model is None:
        model, model_encoder = load_model()
    
    try:
        # Get embedding for the prompt
        embedding = get_embedding(prompt)
        
        # Convert model to index
        try:
            model_idx = model_encoder.transform([model_name])[0]
        except:
            return jsonify({'error': f'Unknown model: {model_name}. Available models: {list(model_encoder.classes_)}'}), 400
        
        # Prepare input tensors
        embedding_tensor = torch.tensor([embedding], dtype=torch.float32)
        model_idx_tensor = torch.tensor([model_idx], dtype=torch.long)
        
        # Make prediction with confidence intervals
        prediction_result = predict_with_confidence(model, embedding_tensor, model_idx_tensor)
        
        # Return the result
        return jsonify({
            'prompt': prompt,
            'model': model_name,
            'predicted_completion_tokens': prediction_result['predicted_tokens'],
            'confidence_interval': prediction_result['confidence_interval'],
            'message': f"95% confidence interval: {prediction_result['confidence_interval']['lower']} to {prediction_result['confidence_interval']['upper']} tokens"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    global model_encoder
    if model_encoder is None:
        model, model_encoder = load_model()
    
    return jsonify({
        'models': list(model_encoder.classes_)
    })

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <head>
            <title>Enhanced Token Prediction API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                pre {
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                .endpoint {
                    margin-bottom: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #3498db;
                }
                .model-info {
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                }
                .confidence {
                    background-color: #f8e8e8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>Enhanced Token Prediction API</h1>
            <p>This API predicts the number of completion tokens for a given prompt and model using an advanced neural network with GLU activation and multiple hidden layers.</p>
            
            <div class="model-info">
                <h3>Model Architecture Highlights:</h3>
                <ul>
                    <li>Gated Linear Unit (GLU) activation functions</li>
                    <li>Multiple hidden layers with deep architecture</li>
                    <li>Layer normalization for improved training stability</li>
                    <li>Trained with 50/20/30 train/validation/test split</li>
                    <li>Uses MSE loss for accurate token prediction</li>
                    <li>Trained for up to 100 epochs with early stopping</li>
                    <li>Optimized with AdamW and learning rate scheduling</li>
                    <li>Streamlined architecture focusing on prompt and model features</li>
                </ul>
            </div>
            
            <div class="confidence">
                <h3>Confidence Intervals:</h3>
                <p>The API provides 95% confidence intervals for token predictions using Monte Carlo Dropout:</p>
                <ul>
                    <li>Multiple forward passes with dropout enabled</li>
                    <li>Statistical analysis of prediction distribution</li>
                    <li>Lower and upper bounds with 95% confidence</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h2>Predict Tokens</h2>
                <p><strong>Endpoint:</strong> POST /predict</p>
                <p><strong>Request Body:</strong></p>
                <pre>
{
    "prompt": "Your prompt text here",
    "model": "model-name"
}
                </pre>
                <p><strong>Response:</strong></p>
                <pre>
{
    "prompt": "Your prompt text here",
    "model": "model-name",
    "predicted_completion_tokens": 123,
    "confidence_interval": {
        "lower": 100,
        "upper": 150,
        "confidence": "95%"
    },
    "message": "95% confidence interval: 100 to 150 tokens"
}
                </pre>
            </div>
            
            <div class="endpoint">
                <h2>List Available Models</h2>
                <p><strong>Endpoint:</strong> GET /models</p>
                <p><strong>Response:</strong></p>
                <pre>
{
    "models": ["gpt-3.5-turbo", "gpt-4o-mini", "o1-mini", "o3-mini", "claude"]
}
                </pre>
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Initialize model before running the app
    initialize()
    
    app.run(debug=True, host='0.0.0.0', port=5001)
