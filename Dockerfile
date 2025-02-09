FROM python:3.9

# Aggiungi queste righe per installare le librerie di sistema necessarie
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /code

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# Copia l'intera directory del progetto
COPY . .
ENV PYTHONUNBUFFERED=1


CMD ["python", "-u", "main.py"]
