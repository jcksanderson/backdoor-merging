from huggingface_hub import snapshot_download

# Replace with the exact model ID you want
MODEL_ID = "meta-llama/Llama-3.1-8B"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir="llama",
    local_dir_use_symlinks=False
)

print("Downloaded model to ./llama/")

