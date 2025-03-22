# Enhanced Token Prediction API

This API predicts the number of completion tokens for a given prompt and model using an advanced neural network. It provides confidence intervals for more reliable estimates.

## Features

-   Predicts completion tokens for a given prompt and model.
-   Provides 95% confidence intervals using Monte Carlo Dropout.
-   Supports multiple models.
-   API endpoints to list available models and make predictions.

## Model Architecture Highlights

-   Gated Linear Unit (GLU) activation functions.
-   Multiple hidden layers with deep architecture.
-   Layer normalization for improved training stability.
-   Trained with 50/20/30 train/validation/test split.
-   Uses MSE loss for accurate token prediction.
-   Trained for up to 100 epochs with early stopping.
-   Optimized with AdamW and learning rate scheduling.
-   Streamlined architecture focusing on prompt and model features.

## Confidence Intervals

The API provides 95% confidence intervals for token predictions using Monte Carlo Dropout:

-   Multiple forward passes with dropout enabled.
-   Statistical analysis of prediction distribution.
-   Lower and upper bounds with 95% confidence.

## Endpoints

-   `POST /predict`: Predicts completion tokens for a given prompt and model.
    -   Request body:

        ```json
        {
            "prompt": "Your prompt text here",
            "model": "model-name"
        }
        ```

    -   Response:

        ```json
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
        ```

-   `GET /models`: Lists available models.
    -   Response:

        ```json
        {
            "models": ["gpt-3.5-turbo", "gpt-4o-mini", "o1-mini", "o3-mini", "claude"]
        }
        ```

## Setup Instructions

1.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Keys**:

    -   Ensure you have an OpenAI API key.
    -   Set the `OPENAI_API_KEY` environment variable in your `.env` file.

        ```properties
        OPENAI_API_KEY=your_openai_api_key
        ```

    -   If you want to use Anthropic models, ensure you have an Anthropic API key.
    -   Set the `ANTHROPIC_API_KEY` environment variable in your `.env` file.

        ```properties
        ANTHROPIC_API_KEY=your_anthropic_api_key
        ```

## Running the Application

1.  **Run the Flask app**:

    ```bash
    python app.py
    ```

    The app will start on `http://0.0.0.0:5001`.

## Accessing the API

To make the API accessible over the internet, you can use **ngrok**.

### Using ngrok

ngrok creates a secure tunnel between your local machine and the ngrok cloud, providing a public URL for accessing your local server.

#### Installation

1.  **Sign up for ngrok**:

    -   Create a free account at [ngrok.com](https://ngrok.com/).

2.  **Install ngrok**:

    -   **macOS**:

        -   Using Homebrew:

            ```bash
            brew install ngrok
            ```

        -   Alternatively, download the macOS package from the ngrok website and follow the installation instructions.

    -   **Windows**:

        -   Download the Windows ZIP file from the ngrok website.
        -   Extract the `ngrok.exe` executable to a directory of your choice.
        -   Add the directory containing `ngrok.exe` to your system's `PATH` environment variable.

#### Configuration

1.  **Authenticate ngrok**:

    -   Get your authtoken from the ngrok dashboard after signing up.
    -   Run the following command in your terminal, replacing `<YOUR_AUTHTOKEN>` with your actual authtoken:

        ```bash
        ngrok config add-authtoken <YOUR_AUTHTOKEN>
        ```

#### Running ngrok

1.  **Start ngrok tunnel**:

    -   To tunnel your Flask app running on port 5001, run:

        ```bash
        ngrok http 5001
        ```

2.  **Access your application**:

    -   ngrok will display a public URL (e.g., `https://your-ngrok-url.ngrok.io`) in the terminal.
    -   Use this URL to access your API from any device connected to the internet.

### Using the API with ngrok

1.  **Start the Flask app**:

    ```bash
    python app.py
    ```

2.  **Start the ngrok tunnel** (as described above).

3.  **Access the API endpoints using the ngrok URL**:

    -   For example, to predict tokens, send a POST request to `https://your-ngrok-url.ngrok.io/predict` with the required JSON payload.
    -   To list available models, send a GET request to `https://your-ngrok-url.ngrok.io/models`.

## Example Usage

### Predict Tokens

Send a POST request to `/predict` with the following JSON payload:

```json
{
    "prompt": "Translate 'hello' to French.",
    "model": "gpt-3.5-turbo"
}
```

The API will return a JSON response with the predicted number of completion tokens and confidence intervals.

### List Available Models

Send a GET request to `/models`. The API will return a JSON response with a list of available models.

```json
{
    "models": ["gpt-3.5-turbo", "gpt-4o-mini", "o1-mini", "o3-mini", "claude"]
}
```

## Additional Notes

-   Ensure your OpenAI API key is correctly set in the `.env` file.
-   The Flask app must be running for ngrok to tunnel the connection.
-   The ngrok URL will change each time you start ngrok unless you have a paid ngrok plan with reserved domains.
