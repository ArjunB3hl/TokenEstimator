import os
import pandas as pd
import glob
import json
from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_openai_response(prompt, model="o3-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            'message': response.choices[0].message.content.strip(),
            'completion_tokens': response.usage.completion_tokens
        }
    except Exception as e:
        print(f"Error getting response: {e}")
        return {'message': f"Error: {str(e)}", 'completion_tokens': 0}


def append_to_json(filename, data):
    if os.path.exists(filename):
        with open(filename, 'r+') as file:
            existing_data = json.load(file)
            existing_data.append(data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(filename, 'w') as file:
            json.dump([data], file, indent=4)


def process_csv_files(directory='.', output_column='response', json_filename='o3-mini-data.json'):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file}...")

            df = pd.read_csv(csv_file)

            if 'prompt' not in df.columns:
                print(f"'prompt' column not found in {csv_file}, skipping...")
                continue

            

            for idx, row in df.iterrows():
                prompt = row['prompt']
                technique = row.get('technique', '')

                if pd.isna(prompt) or not prompt.strip():
                    continue

                print(f"Processing prompt {idx + 1}/{len(df)}")
                response_dict = get_openai_response(prompt)
                output = response_dict['message']
                completion_tokens = response_dict['completion_tokens']

               

                # Save data to JSON
                json_data = {
                    'prompt': prompt,
                    'output': output,
                    'completion_tokens': completion_tokens,
                    'technique': technique
                }

                append_to_json(json_filename, json_data)
                print(f"Processed data saved to json")
            

            
            

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    else:
        process_csv_files()
        print("All CSV files have been processed.")