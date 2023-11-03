# setup repo
# git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt/

# # install the dependencies
pip install -r requirements-all.txt

# # download weights
# python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf
# python scripts/convert_hf_checkpoint.py \
#             --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf

# # Default preparation script
# python scripts/prepare_alpaca.py \
#         --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
#         --max_seq_length 256


# Run finetuning
python finetune/lora.py \
        --checkpoint_dir ./checkpoints/meta-llama/Llama-2-7b-hf/ \
        --data_dir data/alpaca \
        --precision bf16-true
