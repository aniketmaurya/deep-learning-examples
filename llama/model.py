from layers import Transformer, ModelArgs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def map_state_dict(hf_state_dict, custom_model):
    custom_state_dict = custom_model.state_dict()
    mapped_state_dict = {}

    mapped_state_dict['embed_tokens.weight'] = hf_state_dict['model.embed_tokens.weight']
    for i in range(custom_model.params.n_layers):  # Now 16 layers
        prefix = f'layers.{i}'
        hf_prefix = f'model.layers.{i}'
        mapped_state_dict[f'{prefix}.self_attn.q_proj.weight'] = hf_state_dict[f'{hf_prefix}.self_attn.q_proj.weight']
        mapped_state_dict[f'{prefix}.self_attn.k_proj.weight'] = hf_state_dict[f'{hf_prefix}.self_attn.k_proj.weight']
        mapped_state_dict[f'{prefix}.self_attn.v_proj.weight'] = hf_state_dict[f'{hf_prefix}.self_attn.v_proj.weight']
        mapped_state_dict[f'{prefix}.self_attn.o_proj.weight'] = hf_state_dict[f'{hf_prefix}.self_attn.o_proj.weight']
        mapped_state_dict[f'{prefix}.mlp.gate_proj.weight'] = hf_state_dict[f'{hf_prefix}.mlp.gate_proj.weight']
        mapped_state_dict[f'{prefix}.mlp.up_proj.weight'] = hf_state_dict[f'{hf_prefix}.mlp.up_proj.weight']
        mapped_state_dict[f'{prefix}.mlp.down_proj.weight'] = hf_state_dict[f'{hf_prefix}.mlp.down_proj.weight']
        mapped_state_dict[f'{prefix}.input_layernorm.weight'] = hf_state_dict[f'{hf_prefix}.input_layernorm.weight']
        mapped_state_dict[f'{prefix}.post_attention_layernorm.weight'] = hf_state_dict[
            f'{hf_prefix}.post_attention_layernorm.weight']
    mapped_state_dict['norm.weight'] = hf_state_dict['model.norm.weight']
    mapped_state_dict['lm_head.weight'] = hf_state_dict['lm_head.weight']

    custom_state_dict.update(mapped_state_dict)
    custom_model.load_state_dict(custom_state_dict, strict=False)
    return custom_model


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
    # Configure ModelArgs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = ModelArgs(
        device=device,
        dim=2048,
        n_layers=16,  # Match HF's num_hidden_layers
        n_heads=32,
        n_kv_heads=8,  # GQA confirmed
        vocab_size=128256,
        multiple_of=256,
        ffn_dim_multiplier=None,  # Adjust MLP manually
        norm_eps=1e-5,
        rope_theta=500000.0,
        max_batch_size=4,
        max_seq_len=2048
    )

    # Initialize custom model
    llama = Transformer(params).to(device)

    # Load HF model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    # Map and load weights
    llama = map_state_dict(hf_model.state_dict(), llama)
    llama.to(device)  # Ensure model is on the correct device

    # Generate text
    generated_text = generate(llama, tokenizer, "To break eggs properly, you need to", max_length=100, device=device)
    print(generated_text)
