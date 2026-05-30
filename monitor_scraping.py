import argparse
import json
import re
import time
import webbrowser
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_REFRESH_MS = 2000
PRICE_ZERO_LIMIT = 0.0


@dataclass
class MonitorConfig:
    urls_path: Path
    output_path: Path
    log_path: Path
    csv_path: Path
    discovered_path: Path
    refresh_ms: int


def normalize_input_url(raw_url: str) -> str:
    url = raw_url.strip().strip('"').strip("'")
    url = url.replace("\\_", "_").replace("\\&", "&")
    url = url.replace("&amp;", "&")
    if url.startswith("www."):
        url = "https://" + url
    return url


def read_unique_urls(path: Path) -> list[str]:
    if not path.exists():
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        url = normalize_input_url(raw_line)
        if not url or url.startswith("#"):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def read_log_tail(path: Path, max_lines: int = 220) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    except OSError:
        return []


def domain_for_log(url: str) -> str:
    match = re.match(r"^https?://([^/]+)", url)
    return match.group(1) if match else url[:80]


def count_dishes(result: dict[str, Any]) -> int:
    return sum(
        len(category.get("plats", []) or [])
        for category in result.get("categories", []) or []
        if isinstance(category, dict)
    )


def result_has_incomplete_prices(result: dict[str, Any]) -> bool:
    if not result.get("menu_disponible"):
        return False

    dish_count = 0
    for category in result.get("categories", []) or []:
        if not isinstance(category, dict):
            continue
        for dish in category.get("plats", []) or []:
            if not isinstance(dish, dict):
                continue
            dish_count += 1
            try:
                price = float(dish.get("prix_euros", 0))
            except (TypeError, ValueError):
                return True
            if price <= PRICE_ZERO_LIMIT:
                return True

    return dish_count == 0


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size": 0, "mtime": None}
    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


def extract_current_activity(log_lines: list[str]) -> dict[str, str]:
    if not log_lines:
        return {"line": "Aucun log pour le moment.", "index": "", "domain": ""}

    last_line = log_lines[-1]
    index_match = re.search(r"\[(\d+/\d+)\]", last_line)
    domain_match = re.search(r"pour ([^:\s]+)", last_line)
    return {
        "line": last_line,
        "index": index_match.group(1) if index_match else "",
        "domain": domain_match.group(1) if domain_match else "",
    }


def build_status(config: MonitorConfig) -> dict[str, Any]:
    urls = read_unique_urls(config.urls_path)
    results = load_results(config.output_path)
    log_lines = read_log_tail(config.log_path)

    processed_urls = {
        result.get("url_restaurant")
        for result in results
        if result.get("url_restaurant")
    }

    success = [result for result in results if result.get("menu_disponible")]
    failed = [result for result in results if result.get("menu_disponible") is False]
    incomplete = [result for result in results if result_has_incomplete_prices(result)]

    total = len(urls)
    processed = len(processed_urls)
    dish_total = sum(count_dishes(result) for result in results)
    progress = round((processed / total) * 100, 1) if total else 0

    recent_results = []
    for result in results[-250:]:
        url = result.get("url_restaurant", "")
        recent_results.append(
            {
                "url": url,
                "domain": domain_for_log(url),
                "menu": bool(result.get("menu_disponible")),
                "incomplete": result_has_incomplete_prices(result),
                "categories": len(result.get("categories", []) or []),
                "dishes": count_dishes(result),
                "error": result.get("erreur", ""),
            }
        )

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "processed": processed,
        "remaining": max(total - processed, 0),
        "progress": progress,
        "success": len(success),
        "failed": len(failed),
        "incomplete": len(incomplete),
        "dishes": dish_total,
        "current": extract_current_activity(log_lines),
        "recent_results": recent_results,
        "logs": log_lines,
        "files": {
            "urls": file_info(config.urls_path),
            "json": file_info(config.output_path),
            "csv": file_info(config.csv_path),
            "nouveauurl": file_info(config.discovered_path),
            "log": file_info(config.log_path),
        },
    }


def dashboard_html(refresh_ms: int) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scraping menus - live</title>
  <style>
    :root {{
      font-family: Arial, sans-serif;
      color: #172033;
      background: #f4f6fa;
    }}
    body {{ margin: 0; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; margin: 24px 0 10px; }}
    .muted {{ color: #65748a; font-size: 13px; }}
    .actions {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .button {{
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      padding: 0 12px;
      border: 1px solid #cfd8e6;
      border-radius: 6px;
      background: white;
      color: #1a4d85;
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: white;
      border: 1px solid #dfe6f0;
      border-radius: 8px;
      padding: 14px;
      min-height: 72px;
    }}
    .metric span {{ color: #65748a; font-size: 12px; }}
    .metric strong {{ display: block; font-size: 25px; margin-top: 5px; }}
    .progress {{
      height: 14px;
      background: #dbe3ee;
      border-radius: 999px;
      overflow: hidden;
      margin: 16px 0;
    }}
    .progress div {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #1f8a70, #2f7dd1);
      transition: width .25s ease;
    }}
    .current {{
      background: #fff;
      border-left: 4px solid #2f7dd1;
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0;
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #dfe6f0;
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid #eef2f7;
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
    }}
    th {{ background: #e9eef6; color: #37445b; }}
    a {{ color: #185fa7; text-decoration: none; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 76px;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .ok {{ background: #dff6e9; color: #17633a; }}
    .fail {{ background: #ffe1dc; color: #912515; }}
    .warn {{ background: #fff2c7; color: #735000; }}
    pre {{
      background: #111827;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      max-height: 380px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .files {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .file {{
      background: white;
      border: 1px solid #dfe6f0;
      border-radius: 8px;
      padding: 12px;
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .files {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      header {{ display: block; }}
      .actions {{ justify-content: flex-start; margin-top: 12px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Scraping menus - live</h1>
      <div class="muted">Mise a jour auto toutes les {refresh_ms // 1000}s - <span id="updated">chargement</span></div>
    </div>
    <nav class="actions">
      <a class="button" href="/files/menus_lyon.json" target="_blank">JSON</a>
      <a class="button" href="/files/menus_lyon.csv" target="_blank">CSV</a>
      <a class="button" href="/files/nouveauurl.txt" target="_blank">NouveauURL</a>
      <a class="button" href="/files/scrape_menus.log" target="_blank">Logs</a>
    </nav>
  </header>

  <section class="grid">
    <div class="metric"><span>Progression</span><strong id="processed">0/0</strong></div>
    <div class="metric"><span>Restants</span><strong id="remaining">0</strong></div>
    <div class="metric"><span>Menus trouves</span><strong id="success">0</strong></div>
    <div class="metric"><span>A revoir</span><strong id="failed">0</strong></div>
    <div class="metric"><span>Prix incomplets</span><strong id="incomplete">0</strong></div>
    <div class="metric"><span>Plats exportes</span><strong id="dishes">0</strong></div>
  </section>

  <div class="progress"><div id="progressbar"></div></div>

  <section class="current">
    <strong id="current-title">Activite</strong>
    <div id="current-line" class="muted">Chargement...</div>
  </section>

  <h2>Derniers resultats</h2>
  <table>
    <thead>
      <tr><th>Restaurant</th><th>Statut</th><th>Plats</th><th>Categories</th><th>Erreur</th></tr>
    </thead>
    <tbody id="results"></tbody>
  </table>

  <h2>Fichiers</h2>
  <section class="files" id="files"></section>

  <h2>Logs</h2>
  <pre id="logs">Chargement...</pre>
</main>

<script>
const refreshMs = {refresh_ms};

function text(id, value) {{
  document.getElementById(id).textContent = value;
}}

function statusPill(item) {{
  if (item.incomplete) return '<span class="pill warn">Prix</span>';
  if (item.menu) return '<span class="pill ok">OK</span>';
  return '<span class="pill fail">A revoir</span>';
}}

function renderFiles(files) {{
  const root = document.getElementById('files');
  root.innerHTML = Object.entries(files).map(([name, info]) => `
    <div class="file">
      <strong>${{name.toUpperCase()}}</strong><br>
      <span class="muted">${{info.exists ? `${{info.size}} octets` : 'absent'}}</span><br>
      <span class="muted">${{info.mtime || ''}}</span>
    </div>
  `).join('');
}}

async function loadStatus() {{
  const response = await fetch('/api/status?ts=' + Date.now());
  const data = await response.json();

  text('updated', data.generated_at);
  text('processed', `${{data.processed}}/${{data.total}}`);
  text('remaining', data.remaining);
  text('success', data.success);
  text('failed', data.failed);
  text('incomplete', data.incomplete);
  text('dishes', data.dishes);
  document.getElementById('progressbar').style.width = data.progress + '%';

  const currentTitle = data.current.index || data.current.domain
    ? `Activite ${{data.current.index}} ${{data.current.domain}}`
    : 'Activite';
  text('current-title', currentTitle);
  text('current-line', data.current.line);

  const rows = data.recent_results.slice().reverse().map(item => `
    <tr>
      <td><a href="${{item.url}}" target="_blank">${{item.domain || item.url}}</a></td>
      <td>${{statusPill(item)}}</td>
      <td>${{item.dishes}}</td>
      <td>${{item.categories}}</td>
      <td>${{item.error || ''}}</td>
    </tr>
  `).join('');
  document.getElementById('results').innerHTML = rows || '<tr><td colspan="5">Aucun resultat.</td></tr>';
  document.getElementById('logs').textContent = data.logs.join('\\n');
  renderFiles(data.files);
}}

loadStatus().catch(error => {{
  text('current-line', 'Erreur dashboard: ' + error);
}});
setInterval(() => loadStatus().catch(console.error), refreshMs);
</script>
</body>
</html>
"""


def make_handler(config: MonitorConfig):
    class MonitorHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_text(self, body: str, content_type: str = "text/plain", status: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def send_json(self, data: dict[str, Any]) -> None:
            self.send_text(json.dumps(data, ensure_ascii=False), "application/json")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path == "/":
                self.send_text(dashboard_html(config.refresh_ms), "text/html")
                return

            if parsed.path == "/api/status":
                self.send_json(build_status(config))
                return

            if parsed.path == "/files/menus_lyon.json":
                self.serve_file(config.output_path, "application/json")
                return

            if parsed.path == "/files/menus_lyon.csv":
                self.serve_file(config.csv_path, "text/csv")
                return

            if parsed.path == "/files/nouveauurl.txt":
                self.serve_file(config.discovered_path, "text/plain")
                return

            if parsed.path == "/files/scrape_menus.log":
                self.serve_file(config.log_path, "text/plain")
                return

            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            self.send_text("Not found", status=404)

        def serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_text("File not found", status=404)
                return
            try:
                body = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                self.send_text(f"Cannot read file: {escape(str(exc))}", status=500)
                return
            self.send_text(body, content_type)

    return MonitorHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashboard live pour le scraping des menus.")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'ecoute. VPS public: 0.0.0.0")
    parser.add_argument("--port", type=int, default=8787, help="Port HTTP du dashboard.")
    parser.add_argument("--urls", default="urls.txt", help="Fichier d'URLs du scraper.")
    parser.add_argument("--output", default="menus_lyon.json", help="JSON de sortie du scraper.")
    parser.add_argument("--log-file", default="scrape_menus.log", help="Fichier de logs du scraper.")
    parser.add_argument("--csv-output", default="menus_lyon.csv", help="CSV aplati du scraper.")
    parser.add_argument("--discovered-urls", default="nouveauurl.txt", help="URLs menu detectees par le scraper.")
    parser.add_argument("--refresh-ms", type=int, default=DEFAULT_REFRESH_MS, help="Frequence de refresh.")
    parser.add_argument("--open", action="store_true", help="Ouvre le navigateur automatiquement.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MonitorConfig(
        urls_path=Path(args.urls),
        output_path=Path(args.output),
        log_path=Path(args.log_file),
        csv_path=Path(args.csv_output),
        discovered_path=Path(args.discovered_urls),
        refresh_ms=max(args.refresh_ms, 500),
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(config))
    url = f"http://{args.host}:{args.port}/"

    print(f"Dashboard live: {url}")
    print("Ctrl+C pour arreter.")
    if args.host == "0.0.0.0":
        print(f"Depuis un autre poste: http://IP_DU_VPS:{args.port}/")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret du dashboard.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
