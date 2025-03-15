from layers import Transformer, ModelArgs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate(
        model: Transformer,
        tokenizer,
        prompt: str,
        max_length: int = 50,
        device: str = "cpu",
):
    """
    Generate text using the Llama Transformer model.

    Args:
        model (Transformer): The trained Llama model.
        tokenizer: Tokenizer with encode(), decode(), and eos_token_id.
        prompt (str): Initial text to start generation.
        max_length (int): Maximum length of the sequence (prompt + generated).
        device (str): Device to run the model on (e.g., "cuda").

    Returns:
        str: Generated text, including the prompt.
    """
    # Set model to evaluation mode and disable gradients
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt into a list of token IDs
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        # Initialize the sequence with prompt tokens
        sequence = prompt_tokens.copy()

        # Convert prompt to tensor with batch_size=1
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Process the prompt to initialize the cache
        start_pos = 0
        logits = model(tokens, start_pos)

        # Sample the first token after the prompt from the last position's logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        sequence.append(next_token_id)

        # Update start_pos to the length of the prompt for generation
        start_pos = prompt_len

        # Generate subsequent tokens one at a time
        while len(sequence) < max_length:
            # Prepare the next token as a tensor of shape (1, 1)
            next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

            # Forward pass with the new token
            logits = model(next_input, start_pos)

            # Sample the next token from logits (shape: 1, 1, vocab_size)
            next_token_id = torch.argmax(logits[:, 0, :], dim=-1).item()
            sequence.append(next_token_id)

            # Increment start_pos for the next token
            start_pos += 1

            # Stop if EOS token is generated
            if next_token_id == tokenizer.eos_token_id:
                break

        # Decode the full sequence (prompt + generated) to text
        generated_text = tokenizer.decode(sequence)
        return generated_text


if __name__ == '__main__':
    params = ModelArgs()
    params.dim = 2048
    with torch.device(params.device):
        llama = Transformer(params)
        # print(llama)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    llama.load_state_dict(model.state_dict(), strict=False)
    llama = llama.eval()

    generated_text = generate(llama, tokenizer, "How to", max_length=100, device=params.device)
    print(generated_text)
