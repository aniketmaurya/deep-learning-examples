from layers import Transformer, ModelArgs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate(model, tokenizer, prompt, max_length=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        prompt_tokens = tokenizer.encode(prompt)
        sequence = prompt_tokens.copy()
        tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        logits = model(tokens, start_pos=0)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        sequence.append(next_token_id)

        start_pos = len(prompt_tokens)
        while len(sequence) < max_length:
            next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            logits = model(next_input, start_pos)
            next_token_id = torch.argmax(logits[:, 0, :], dim=-1).item()
            sequence.append(next_token_id)
            start_pos += 1
            if next_token_id == tokenizer.eos_token_id:
                break

        return tokenizer.decode(sequence)


if __name__ == '__main__':
    params = ModelArgs()
    params.dim = 2048
    with torch.device(params.device):
        llama = Transformer(params)
        print(llama)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    llama.load_state_dict(model.state_dict(), strict=False)
    llama = llama.eval()

    generated_text = generate(llama, tokenizer, "How to break eggs properly ", max_length=100, device=params.device)
    print(generated_text)
