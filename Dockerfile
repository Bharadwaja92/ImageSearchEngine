FROM python:3.12.3-slim

WORKDIR /app

# Install system packages (including curl)
RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import CLIPProcessor, CLIPModel; \
               CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
               CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

COPY . .

RUN chmod +x start.sh
ENTRYPOINT ["./start.sh"]