# LangChain Project

This repository contains a LangChain-based documentation ingestion pipeline.

## Overview

- `langchain-doc-index.py`: Main script to crawl, extract, split, and index documentation content.
- `.env`: Environment variables for API keys and settings.
- `.gitignore`: Standard Python ignores, including `.venv` and `.env`.
- `logger.py`: Local logger helper used by the pipeline.

## Setup

1. Activate the virtual environment:
```powershell
cd "C:\Udemy LangChain"
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies if needed:
```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## GitHub / Git

To connect to your GitHub repo:
```powershell
cd "C:\Udemy LangChain"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/kamal8892/langchain-project.git
git push -u origin main
```

When Git prompts for credentials, use:
- `Username`: `kamal8892`
- `Password`: your GitHub personal access token

## Notes

- Do not commit secrets or tokens.
- The `.env` file should remain private and excluded from Git.
- If you use cached Git credentials, clear them before pushing.
