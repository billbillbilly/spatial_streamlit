#!/bin/sh
conda activate llmApp
ollama pull llama3.2-vision
python app.py