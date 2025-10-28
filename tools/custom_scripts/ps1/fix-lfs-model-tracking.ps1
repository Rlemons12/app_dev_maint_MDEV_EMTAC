Write-Host "=== Re-indexing large AI model files under Git LFS ==="

# Ensure we’re at the repo root
Set-Location "E:\emtac"

# Stage the .gitattributes file just in case
git add .gitattributes

# Remove any cached versions (if they exist)
$paths = @(
    "models/llm/TinyLlama_1_1B/model.safetensors",
    "models/llm/TinyLlama_1_1B/smoketest_results/checkpoint-2/model.safetensors",
    "models/llm/TinyLlama_1_1B/smoketest_results/checkpoint-2/optimizer.pt",
    "models/llm/Qwen2.5-3B-Instruct/model-00003-of-00003.safetensors",
    "models/llm/apple_OpenELM-1_1B-Instruct/model.safetensors",
    "models/image/openai_clip-vit-base-patch32/model.safetensors",
    "models/llm/google_gemma-2-2b-it/model-00002-of-00002.safetensors"
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Host "Removing cached version of $p"
        git rm --cached "$p" 2>$null
    } else {
        Write-Host "Skipping missing file: $p"
    }
}

# Re-add all model files so LFS picks them up
git add models/

# Commit
git commit -m "Move all large model weights to Git LFS"

# Verify
git lfs ls-files
Write-Host "`nAll large model files should now be managed by LFS."
