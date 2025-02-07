#!/bin/sh
conda init bash
conda activate llmApp
ollama pull llama3.2-vision
streamlit run app.py