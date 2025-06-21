FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04


# Installa Python e dipendenze di sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copia e installa requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Installa PyTorch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Installa dipendenze base
RUN pip3 install \
    supervision \
    opencv-python \
    transformers \
    huggingface_hub \
    numpy \
    pillow

# Installa SAM2 dal repository (come funzionava prima)
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /tmp/sam2 && \
    cd /tmp/sam2 && \
    pip3 install -e .

# Prova GroundingDINO precompilato
RUN pip3 install groundingdino-py

# Crea directory per i modelli
RUN mkdir -p /code/models

# Scarica modelli GroundingDINO
RUN cd /code/models && \
    wget -O groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && \
    wget -O GroundingDINO_SwinT_OGC.py \
        https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# Scarica modelli SAM2
RUN cd /code/models && \
    wget -O sam2_hiera_base_plus.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Variabili d'ambiente
ENV GROUNDING_DINO_CONFIG_PATH=/code/models/GroundingDINO_SwinT_OGC.py
ENV GROUNDING_DINO_CHECKPOINT_PATH=/code/models/groundingdino_swint_ogc.pth
ENV SAM2_CHECKPOINT_PATH=/code/models/sam2_hiera_base_plus.pt
ENV PYTHONUNBUFFERED=1

COPY . .

CMD ["python3", "main.py"]