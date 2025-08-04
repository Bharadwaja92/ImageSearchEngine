# Image Search Application

## Overview

This project is an Image Search Engine that allows users to input a natural language query and retrieve the most relevant images from a dataset using CLIP embeddings and Qdrant for vector search. The application has two interfaces:
- A FastAPI backend for programmatic access.
- A Streamlit frontend for user-friendly interaction.

## Features

- CLIP-based text-to-image retrieval.
- FastAPI for backend inference.
- Streamlit for a simple UI.
- Qdrant for storing and searching image embeddings.
- Dockerized setup with `docker-compose`.
- Test coverage up to **92%** for the core engine.

---

## Folder Structure

```
.
├── app/                        # Core logic (ImageSearchEngine, ingestion, etc.)
├── config/                     # Configurations (e.g., YAML, Prompt Templates)
├── images/                     # Retrieved/downloaded images
├── tests/                      # Test suite with pytest
├── Dockerfile
├── docker-compose.yml
├── start.sh                    # Starts Streamlit, FastAPI, and ingestion
├── Streamlit_App.py           # Streamlit interface
├── FastAPI_page.py            # FastAPI interface
```

---

## Test Coverage

We use `pytest` and `pytest-cov` to test the engine. Run the following command to generate a coverage report:

```bash
pytest tests/test.py --cov=app --cov-report=term --cov-report=html
```

The current test coverage is **92%**, and the HTML report will be stored in `htmlcov/index.html`.

---

## Image Ingestion Process

Two ingestion methods are provided:

### 1. From Image Files (Download from Flickr)
```bash
python download_images.py
```
This script uses a pre-generated list of image URLs (`photos_url.csv`) and downloads 2000 images into the `images/` directory.

### 2. From Qdrant Snapshot
```bash
python app/IngestDataFromSnapshot.py
```
This restores the image embeddings into Qdrant using a snapshot file (`QdrantSnapshot.snapshot`) located in the `app/` folder.

---

## How to Run

### 1. Build and Run with Docker Compose

```bash
docker-compose up --build
```

### 2. Access Interfaces

- **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Environment Configuration

The `config/creds.yaml` file includes necessary endpoints and prompt path:

```yaml
QDRANT_URL: http://qdrant-db:6333
LLM_URL: http://ollama:11434/api/generate
QDRANT_COLLECTION_NAME: images
PROMPT_TEMPLATE_PATH: config/ImageMatchPrompt.txt
```

---

## Notes

- LLM-based explanations are currently disabled in the Docker image to reduce size. This feature works on local systems with the model pulled manually using Ollama.
- If you wish to enable it, ensure Ollama is running with the `llama3` model.

