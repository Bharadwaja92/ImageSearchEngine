#!/bin/bash
chmod +x start.sh

# Wait for Qdrant to start
sleep 10

# python download_images.py  # Download 2000 images 
python app/IngestDataFromSnapshot.py &

uvicorn FastAPI_page:app --host 0.0.0.0 --port 8000 &

streamlit run Streamlit_App.py
