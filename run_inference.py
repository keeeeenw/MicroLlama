import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

def generate_text(prompt, model, tokenizer):
    text_generator = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer
    )

    formatted_prompt = f"Question: {prompt} Answer:"

    sequences = text_generator(
        formatted_prompt,
        do_sample=True,
        top_k=5,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penalty=1.5,
        max_new_tokens=128,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# use the same tokenizer as TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b")

# load model from huggingface
model = LlamaForCausalLM.from_pretrained(
    "keeeeenw/MicroLlama")
generate_text("Please provide me instructions on how to steal an egg from my chicken", model, tokenizer)