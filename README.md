# Restaurant Menu Scraper

Scraper Python pour extraire des menus de restaurants via Zyte + OpenAI, avec interface web locale.

## Installation

```bash
pip install -r requirements.txt
```

Copiez `.env.example` en `.env`, puis renseignez :

```env
ZYTE_API_KEY=...
OPENAI_API_KEY=...
```

## Interface Web

```bash
python server.py
```

La page s'ouvre automatiquement sur `http://127.0.0.1:8787/`.

Sur un VPS :

```bash
python server.py --host 0.0.0.0 --port 8787 --no-open
```

Puis ouvrez `http://IP_DU_VPS:8787/`.

Depuis l'interface, vous pouvez :

- importer/coller une liste d'URLs ;
- choisir le nombre de workers ;
- lancer, arreter ou relancer le scraping ;
- suivre les logs et resultats en temps reel ;
- telecharger `menus_lyon.json`, `menus_lyon.csv` et `nouveauurl.txt`.

## CLI Directe

```bash
python scrape_menus.py --retry-failed --retry-incomplete --workers 3 --max-menu-pages 4
```

## Notes

- Ne publiez jamais votre fichier `.env`.
- Commencez avec `--workers 3`, puis augmentez seulement si vos quotas Zyte/OpenAI suivent.
