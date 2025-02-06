FROM python:3.9

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia i requisiti del progetto nella directory di lavoro
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice sorgente nel container
COPY . .

# Comando per avviare il job-executor (sostituisci con il comando giusto per il tuo progetto)
CMD ["python", "main.py"]