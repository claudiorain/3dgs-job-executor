FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# --- system deps + Node.js 20 LTS ---
RUN set -eux; \
    for i in 1 2 3 4 5; do \
      apt-get update && break || (echo "apt update retry $i/5" && sleep 5); \
    done; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip git wget curl ca-certificates gnupg \
      libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install supervision opencv-python transformers huggingface_hub numpy pillow

# SAM2
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /tmp/sam2 \
 && cd /tmp/sam2 && pip3 install -e .

# GroundingDINO
RUN pip3 install groundingdino-py

# Modelli
RUN mkdir -p /code/models \
 && cd /code/models \
 && wget -O groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
 && wget -O GroundingDINO_SwinT_OGC.py https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
 && wget -O sam2_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

ENV GROUNDING_DINO_CONFIG_PATH=/code/models/GroundingDINO_SwinT_OGC.py
ENV GROUNDING_DINO_CHECKPOINT_PATH=/code/models/groundingdino_swint_ogc.pth
ENV SAM2_CHECKPOINT_PATH=/code/models/sam2_hiera_base_plus.pt
ENV PYTHONUNBUFFERED=1

# --- installa la libreria JS per la conversione KSPLAT ---
# (la teniamo localmente così non usi npx ogni volta)
# --- Clona direttamente GaussianSplats3D e builda ---
# Node già presente
WORKDIR /code

# Clona e builda il repo JS
RUN git clone --depth=1 https://github.com/mkkellogg/GaussianSplats3D.git /code/gaussian-splats-3d \
 && cd /code/gaussian-splats-3d \
 && npm ci \
 && npm run build


# Se vuoi, copia util/create-ksplat.js in una posizione comoda
# Crea la cartella /code/util e copia create-ksplat.js
RUN mkdir -p /code/util && \
    cp /code/gaussian-splats-3d/util/create-ksplat.js /code/util/create-ksplat.js

COPY . .

CMD ["python3", "main.py"]
