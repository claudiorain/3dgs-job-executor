FROM python:3.9

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia l'intera directory del progetto
COPY . .
ENV PYTHONUNBUFFERED=1


CMD ["python", "-u", "main.py"]
