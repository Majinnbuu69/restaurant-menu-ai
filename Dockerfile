FROM python:3.12-slim

WORKDIR /app

# Dépendances système pour pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY server.py scrape_menus.py monitor_scraping.py ./

# Dossier persistant pour les données (monté en volume)
RUN mkdir -p /data

# Variables d'environnement : chemins vers /data (volume persistant)
ENV MENU_OUTPUT_PATH=/data/menus.json
ENV MENU_CSV_PATH=/data/menus.csv
ENV MENU_LOG_PATH=/data/scrape_menus.log
ENV MENU_DISCOVERED_PATH=/data/nouveauurl.txt
ENV MENU_URLS_PATH=/data/urls.txt

EXPOSE 8787

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8787", "--no-open"]
