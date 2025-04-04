from huggingface_hub import InferenceClient
from streaming_stt_nemo import Model
import random
import torch


# Random seed generator
def randomize_seed_fn(seed: int) -> int:
    seed = random.randint(0, 999999)
    return seed


# Function to generate AI response using the selected model
def call_llama(prompt, seed=42):
    seed = int(randomize_seed_fn(seed))
    generator = torch.Generator().manual_seed(seed)

    client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2")

    prompt = [
        {"role": "user", "content": f"{prompt}"}
    ]

    output = ""
    try:
        for token in client.chat_completion(prompt, max_tokens=200, stream=True):
            if token.choices and len(token.choices) > 0:
                delta_content = token.choices[0].delta.content
                if delta_content:
                    output += delta_content
    except Exception as e:
        raise RuntimeError(f"Error during text generation: {e}")

    return output
