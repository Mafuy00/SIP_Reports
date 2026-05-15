# Hugging Face Token Setup Guide

## What Changed
The RAG pipeline now uses a `.env` file to safely manage your Hugging Face (HF) token instead of hardcoding it in the Python code.

## Setup Steps

### 1. Get Your Hugging Face Token
- Visit: https://huggingface.co/settings/tokens
- Create a new **Read** token (you don't need write permissions for model downloads)
- Copy the token to your clipboard

### 2. Update the `.env` File
Edit the `.env` file in this directory (`SIPReports_ToYunLin/.env`) and replace:
```
HF_TOKEN=your_hf_token_here
```
with your actual token:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. (Optional) Set Ollama Model
If you want to use a specific Ollama model, update:
```
OLLAMA_MODEL=phi3
```
Available options: `phi3`, `mistral`, `gemma2:2b`, `llama3.2`, `llama3`

## Why This Matters
- **Security**: Your token is not stored in version control or code files
- **Flexibility**: You can have different tokens for different environments
- **Rate Limits**: Authenticated requests get higher rate limits from Hugging Face

## Troubleshooting

### "HF_TOKEN not set in .env file"
- Make sure the `.env` file is in the `SIPReports_ToYunLin/` directory
- Check that `HF_TOKEN=` is on its own line with no extra spaces
- Restart Python/Jupyter after updating the `.env` file

### "HF login failed"
- Your token may be invalid or expired - get a new one from https://huggingface.co/settings/tokens
- Make sure you copied the entire token string correctly
- The pipeline will continue with limited access (no authentication)

### ".env file not found"
- Create a new file called `.env` in the `SIPReports_ToYunLin/` directory
- Use the template provided in this directory

## Testing
After setting your token, the pipeline will:
1. Skip validation if token is a placeholder (`your_hf_token_here`)
2. Log in silently if token is valid
3. Continue with a warning if login fails (but with reduced rate limits)

Run the pipeline normally - no special commands needed!
