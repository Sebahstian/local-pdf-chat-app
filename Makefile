.PHONY: setup pull-models run test

setup:            ## Install Python dependencies (requires uv)
	uv sync

pull-models:      ## Download the chat + embedding models (requires ollama)
	ollama pull llama3.2:3b
	ollama pull nomic-embed-text

run:              ## Start the app at http://localhost:8501
	uv run streamlit run app.py

test:             ## Run the test suite (no Ollama needed)
	uv run pytest
