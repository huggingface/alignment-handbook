import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference(
    model_name_or_path: str,
    word: str,
    context: str,
    tokenizer_name_or_path: str = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    torch_dtype=torch.bfloat16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Run chat inference on a causal language model.
    
    Args:
        model_name_or_path: Path or name of the model to load
        word: The word to disambiguate
        context: Context containing the word
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        torch_dtype: Torch dtype for model
        device: Device to run inference on
    """
    logger.info(f"Loading model {model_name_or_path}")
    
    # Load model and tokenizer
    if tokenizer_name_or_path is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # Check if model path contains checkpoint directory
    if os.path.exists(os.path.join(model_name_or_path, "checkpoint-4")):
        model_name_or_path = os.path.join(model_name_or_path, "checkpoint-4")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    # Read system prompt
    with open("/workspace/alignment-handbook/system_prompt1.txt", "r") as f:
        system_prompt = f.read().strip()
    
    # Construct chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{word} | {context}"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    
    # Decode and extract assistant response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract text between <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|> and next <|START_OF_TURN_TOKEN|>
    assistant_parts = full_response.split("<|START_OF_TURN_TOKEN|>")
    for part in assistant_parts:
        if "<|CHATBOT_TOKEN|>" in part:
            assistant_response = part.split("<|CHATBOT_TOKEN|>")[1].strip()
            return assistant_response
    
    # Fallback if pattern not found
    return full_response

if __name__ == "__main__":
    # Example usage
    model_path = "/workspace/output/wsd-temporal-exp1"
    tokenizer_path = "CohereLabs/c4ai-command-r7b-arabic-02-2025"
    word = input("Enter a word: ")
    context = input("Enter a context: ")
    
    response = run_inference(
        model_name_or_path=model_path,
        tokenizer_name_or_path=tokenizer_path,
        word=word,
        context=context
    )
    print(f"\nAssistant response:\n{response}")
