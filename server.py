import argparse
import json
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import monitor_scraping


DEFAULT_PORT = 8787
SCRAPER_SCRIPT = "scrape_menus.py"
SERVER_STDOUT_LOG = "server_scraper_stdout.log"


@dataclass
class AppConfig:
    host: str
    port: int
    urls_path: Path = Path("urls.txt")
    output_path: Path = Path("menus_lyon.json")
    csv_path: Path = Path("menus_lyon.csv")
    log_path: Path = Path("scrape_menus.log")
    discovered_path: Path = Path("nouveauurl.txt")
    refresh_ms: int = 2000


@dataclass
class ScraperState:
    process: subprocess.Popen | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_returncode: int | None = None
    last_command: list[str] = field(default_factory=list)
    last_error: str | None = None


STATE = ScraperState()
STATE_LOCK = Lock()


def normalize_urls_text(raw_text: str) -> str:
    urls: list[str] = []
    seen: set[str] = set()

    for raw_line in raw_text.splitlines():
        url = monitor_scraping.normalize_input_url(raw_line)
        if not url or url.startswith("#"):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    return "\n".join(urls) + ("\n" if urls else "")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8", errors="ignore")
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Le body JSON doit etre un objet.")
    return data


def as_bool(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_int(data: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def as_float(data: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def build_scraper_command(config: AppConfig, options: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        SCRAPER_SCRIPT,
        "--urls",
        str(config.urls_path),
        "--output",
        str(config.output_path),
        "--csv-output",
        str(config.csv_path),
        "--log-file",
        str(config.log_path),
        "--discovered-urls",
        str(config.discovered_path),
        "--workers",
        str(as_int(options, "workers", 3, 1, 12)),
        "--max-menu-pages",
        str(as_int(options, "max_menu_pages", 4, 1, 10)),
        "--zyte-retries",
        str(as_int(options, "zyte_retries", 3, 0, 10)),
        "--zyte-timeout",
        str(as_int(options, "zyte_timeout", 60, 15, 180)),
        "--openai-timeout",
        str(as_int(options, "openai_timeout", 90, 15, 180)),
        "--sleep",
        str(as_float(options, "sleep", 0.2, 0.0, 10.0)),
    ]

    if as_bool(options, "retry_failed", True):
        command.append("--retry-failed")
    if as_bool(options, "retry_incomplete", True):
        command.append("--retry-incomplete")
    if as_bool(options, "overwrite", False):
        command.append("--overwrite")
    if as_bool(options, "keep_duplicates", False):
        command.append("--keep-duplicates")
    if as_bool(options, "no_interactive_expand", False):
        command.append("--no-interactive-expand")
    if as_bool(options, "no_network_capture", False):
        command.append("--no-network-capture")

    zyte_ip_type = str(options.get("zyte_ip_type", "") or "").strip()
    if zyte_ip_type in {"datacenter", "residential"}:
        command.extend(["--zyte-ip-type", zyte_ip_type])

    zyte_geolocation = str(options.get("zyte_geolocation", "") or "").strip()
    if zyte_geolocation:
        command.extend(["--zyte-geolocation", zyte_geolocation])

    return command


def refresh_process_state() -> None:
    with STATE_LOCK:
        if STATE.process is None:
            return
        returncode = STATE.process.poll()
        if returncode is None:
            return
        STATE.last_returncode = returncode
        STATE.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE.process = None


def is_running() -> bool:
    refresh_process_state()
    with STATE_LOCK:
        return STATE.process is not None and STATE.process.poll() is None


def start_scraper(config: AppConfig, options: dict[str, Any]) -> dict[str, Any]:
    refresh_process_state()
    with STATE_LOCK:
        if STATE.process is not None and STATE.process.poll() is None:
            return {"ok": False, "error": "Un scraping est deja en cours."}

        if not Path(SCRAPER_SCRIPT).exists():
            return {"ok": False, "error": f"Script introuvable: {SCRAPER_SCRIPT}"}

        command = build_scraper_command(config, options)
        stdout_file = open(SERVER_STDOUT_LOG, "a", encoding="utf-8", errors="ignore")
        stdout_file.write("\n\n=== START " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
        stdout_file.write(" ".join(command) + "\n")
        stdout_file.flush()

        try:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            stdout_file.close()
            STATE.last_error = str(exc)
            return {"ok": False, "error": str(exc)}

        STATE.process = process
        STATE.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE.finished_at = None
        STATE.last_returncode = None
        STATE.last_command = command
        STATE.last_error = None

        return {"ok": True, "pid": process.pid, "command": command}


def stop_scraper() -> dict[str, Any]:
    refresh_process_state()
    with STATE_LOCK:
        process = STATE.process
        if process is None or process.poll() is not None:
            STATE.process = None
            return {"ok": True, "message": "Aucun scraping en cours."}

        pid = process.pid
        process.terminate()

    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=8)

    with STATE_LOCK:
        STATE.last_returncode = process.returncode
        STATE.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE.process = None

    return {"ok": True, "message": f"Scraping arrete (pid {pid})."}


def app_status(config: AppConfig) -> dict[str, Any]:
    refresh_process_state()
    monitor_config = monitor_scraping.MonitorConfig(
        urls_path=config.urls_path,
        output_path=config.output_path,
        log_path=config.log_path,
        csv_path=config.csv_path,
        discovered_path=config.discovered_path,
        refresh_ms=config.refresh_ms,
    )
    status = monitor_scraping.build_status(monitor_config)

    with STATE_LOCK:
        running = STATE.process is not None and STATE.process.poll() is None
        status["job"] = {
            "running": running,
            "pid": STATE.process.pid if running and STATE.process else None,
            "started_at": STATE.started_at,
            "finished_at": STATE.finished_at,
            "last_returncode": STATE.last_returncode,
            "last_command": STATE.last_command,
            "last_error": STATE.last_error,
        }

    return status


def dashboard_html(refresh_ms: int) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Restaurant Menu Scraper</title>
  <style>
    :root {{ font-family: Arial, sans-serif; color: #172033; background: #f4f6fa; }}
    body {{ margin: 0; }}
    main {{ max-width: 1260px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: 0; }}
    h2 {{ margin: 22px 0 10px; font-size: 18px; }}
    label {{ display: block; font-size: 12px; color: #5f6d82; font-weight: 700; margin-bottom: 5px; }}
    input, textarea, select {{
      width: 100%; box-sizing: border-box; border: 1px solid #cdd7e5; border-radius: 6px;
      background: white; padding: 9px 10px; font-size: 14px; color: #172033;
    }}
    textarea {{ min-height: 170px; resize: vertical; font-family: Consolas, monospace; }}
    button, .button {{
      min-height: 38px; border: 1px solid #cdd7e5; border-radius: 6px; background: white;
      color: #174f88; font-size: 14px; font-weight: 700; padding: 0 14px; cursor: pointer;
      text-decoration: none; display: inline-flex; align-items: center; justify-content: center;
    }}
    button.primary {{ background: #1f7a5f; border-color: #1f7a5f; color: white; }}
    button.danger {{ background: #b83220; border-color: #b83220; color: white; }}
    button.warning {{ background: #f0b429; border-color: #f0b429; color: #2d2500; }}
    button:disabled {{ opacity: .55; cursor: not-allowed; }}
    .muted {{ color: #65748a; font-size: 13px; }}
    .layout {{ display: grid; grid-template-columns: minmax(320px, 420px) 1fr; gap: 16px; align-items: start; }}
    .panel {{ background: white; border: 1px solid #dfe6f0; border-radius: 8px; padding: 16px; }}
    .form-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .checks {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    .check {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #243047; }}
    .check input {{ width: auto; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .links {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ background: white; border: 1px solid #dfe6f0; border-radius: 8px; padding: 13px; min-height: 70px; }}
    .metric span {{ color: #65748a; font-size: 12px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 5px; }}
    .progress {{ height: 14px; background: #dbe3ee; border-radius: 999px; overflow: hidden; margin: 14px 0; }}
    .progress div {{ height: 100%; width: 0%; background: linear-gradient(90deg, #1f8a70, #2f7dd1); transition: width .25s ease; }}
    .current {{ border-left: 4px solid #2f7dd1; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dfe6f0; border-radius: 8px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #eef2f7; padding: 9px 10px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #e9eef6; color: #37445b; }}
    a {{ color: #185fa7; text-decoration: none; }}
    .pill {{ display: inline-flex; min-width: 72px; justify-content: center; border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 700; }}
    .ok {{ background: #dff6e9; color: #17633a; }}
    .fail {{ background: #ffe1dc; color: #912515; }}
    .warn {{ background: #fff2c7; color: #735000; }}
    pre {{ background: #111827; color: #e5e7eb; border-radius: 8px; padding: 14px; max-height: 360px; overflow: auto; font-size: 12px; line-height: 1.45; white-space: pre-wrap; }}
    .status-line {{ padding: 10px 12px; border-radius: 8px; background: #eef4ff; margin-top: 10px; font-size: 13px; }}
    @media (max-width: 1040px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      header {{ display: block; }}
      .links {{ justify-content: flex-start; margin-top: 12px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Restaurant Menu Scraper</h1>
      <div class="muted">Controle scraper + monitoring live - refresh {refresh_ms // 1000}s - <span id="updated">chargement</span></div>
    </div>
    <nav class="links">
      <a class="button" href="/files/menus_lyon.json" target="_blank">JSON</a>
      <a class="button" href="/files/menus_lyon.csv" target="_blank">CSV</a>
      <a class="button" href="/files/nouveauurl.txt" target="_blank">NouveauURL</a>
      <a class="button" href="/files/scrape_menus.log" target="_blank">Logs</a>
    </nav>
  </header>

  <section class="layout">
    <div class="panel">
      <h2>URLs</h2>
      <label for="urlFile">Importer un .txt</label>
      <input id="urlFile" type="file" accept=".txt,text/plain">
      <label for="urlsText" style="margin-top:10px">URLs a scraper</label>
      <textarea id="urlsText" placeholder="Une URL par ligne"></textarea>
      <div class="actions">
        <button id="saveUrls">Enregistrer les URLs</button>
        <button id="reloadUrls">Recharger urls.txt</button>
      </div>

      <h2>Options</h2>
      <div class="form-grid">
        <div><label>Workers</label><input id="workers" type="number" min="1" max="12" value="3"></div>
        <div><label>Pages menu</label><input id="max_menu_pages" type="number" min="1" max="10" value="4"></div>
        <div><label>Zyte retries</label><input id="zyte_retries" type="number" min="0" max="10" value="3"></div>
        <div><label>Zyte timeout</label><input id="zyte_timeout" type="number" min="15" max="180" value="60"></div>
        <div><label>OpenAI timeout</label><input id="openai_timeout" type="number" min="15" max="180" value="90"></div>
        <div><label>Pause jobs</label><input id="sleep" type="number" min="0" max="10" step="0.1" value="0.2"></div>
        <div><label>Zyte IP</label><select id="zyte_ip_type"><option value="">Auto</option><option value="datacenter">Datacenter</option><option value="residential">Residential</option></select></div>
        <div><label>Geo</label><input id="zyte_geolocation" value="FR"></div>
      </div>
      <div class="checks">
        <label class="check"><input id="retry_failed" type="checkbox" checked> Retry failed</label>
        <label class="check"><input id="retry_incomplete" type="checkbox" checked> Retry incomplete</label>
        <label class="check"><input id="overwrite" type="checkbox"> Overwrite</label>
        <label class="check"><input id="keep_duplicates" type="checkbox"> Garder doublons exacts</label>
        <label class="check"><input id="no_interactive_expand" type="checkbox"> Sans clic onglets</label>
        <label class="check"><input id="no_network_capture" type="checkbox"> Sans capture reseau</label>
      </div>
      <div class="actions">
        <button id="start" class="primary">Lancer</button>
        <button id="abort" class="danger">Abort</button>
        <button id="restart" class="warning">Restart</button>
      </div>
      <div id="message" class="status-line">Pret.</div>
    </div>

    <div>
      <section class="grid">
        <div class="metric"><span>Job</span><strong id="job">Idle</strong></div>
        <div class="metric"><span>Progression</span><strong id="processed">0/0</strong></div>
        <div class="metric"><span>Restants</span><strong id="remaining">0</strong></div>
        <div class="metric"><span>Menus trouves</span><strong id="success">0</strong></div>
        <div class="metric"><span>A revoir</span><strong id="failed">0</strong></div>
        <div class="metric"><span>Plats exportes</span><strong id="dishes">0</strong></div>
      </section>
      <div class="progress"><div id="progressbar"></div></div>
      <section class="panel current">
        <strong id="current-title">Activite</strong>
        <div id="current-line" class="muted">Chargement...</div>
      </section>

      <h2>Derniers resultats</h2>
      <table>
        <thead><tr><th>Restaurant</th><th>Statut</th><th>Plats</th><th>Categories</th><th>Erreur</th></tr></thead>
        <tbody id="results"></tbody>
      </table>

      <h2>Logs</h2>
      <pre id="logs">Chargement...</pre>
    </div>
  </section>
</main>

<script>
const refreshMs = {refresh_ms};
const fields = ['workers','max_menu_pages','zyte_retries','zyte_timeout','openai_timeout','sleep','zyte_ip_type','zyte_geolocation'];
const checks = ['retry_failed','retry_incomplete','overwrite','keep_duplicates','no_interactive_expand','no_network_capture'];

function text(id, value) {{ document.getElementById(id).textContent = value; }}
function msg(value) {{ text('message', value); }}
function getOptions() {{
  const data = {{}};
  for (const id of fields) data[id] = document.getElementById(id).value;
  for (const id of checks) data[id] = document.getElementById(id).checked;
  return data;
}}
async function postJson(url, data) {{
  const response = await fetch(url, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(data || {{}})
  }});
  const body = await response.json();
  if (!response.ok || body.ok === false) throw new Error(body.error || body.message || 'Erreur serveur');
  return body;
}}
function pill(item) {{
  if (item.incomplete) return '<span class="pill warn">Prix</span>';
  if (item.menu) return '<span class="pill ok">OK</span>';
  return '<span class="pill fail">A revoir</span>';
}}
async function loadUrls() {{
  const res = await fetch('/api/urls?ts=' + Date.now());
  const data = await res.json();
  document.getElementById('urlsText').value = data.urls_text || '';
}}
async function loadStatus() {{
  const response = await fetch('/api/status?ts=' + Date.now());
  const data = await response.json();
  text('updated', data.generated_at);
  text('job', data.job.running ? 'Running' : 'Idle');
  text('processed', `${{data.processed}}/${{data.total}}`);
  text('remaining', data.remaining);
  text('success', data.success);
  text('failed', data.failed);
  text('dishes', data.dishes);
  document.getElementById('progressbar').style.width = data.progress + '%';
  document.getElementById('start').disabled = data.job.running;
  document.getElementById('abort').disabled = !data.job.running;
  const currentTitle = data.current.index || data.current.domain ? `Activite ${{data.current.index}} ${{data.current.domain}}` : 'Activite';
  text('current-title', currentTitle);
  text('current-line', data.current.line);
  document.getElementById('results').innerHTML = data.recent_results.slice().reverse().map(item => `
    <tr>
      <td><a href="${{item.url}}" target="_blank">${{item.domain || item.url}}</a></td>
      <td>${{pill(item)}}</td><td>${{item.dishes}}</td><td>${{item.categories}}</td><td>${{item.error || ''}}</td>
    </tr>
  `).join('') || '<tr><td colspan="5">Aucun resultat.</td></tr>';
  document.getElementById('logs').textContent = data.logs.join('\\n');
}}
document.getElementById('urlFile').addEventListener('change', async (event) => {{
  const file = event.target.files[0];
  if (!file) return;
  document.getElementById('urlsText').value = await file.text();
}});
document.getElementById('saveUrls').addEventListener('click', async () => {{
  try {{
    const body = await postJson('/api/urls', {{ urls_text: document.getElementById('urlsText').value }});
    msg(`URLs enregistrees: ${{body.count}}`);
    await loadStatus();
  }} catch (error) {{ msg(error.message); }}
}});
document.getElementById('reloadUrls').addEventListener('click', () => loadUrls().catch(error => msg(error.message)));
document.getElementById('start').addEventListener('click', async () => {{
  try {{ const body = await postJson('/api/start', getOptions()); msg(`Scraping lance pid=${{body.pid}}`); await loadStatus(); }}
  catch (error) {{ msg(error.message); }}
}});
document.getElementById('abort').addEventListener('click', async () => {{
  try {{ const body = await postJson('/api/abort', {{}}); msg(body.message || 'Arrete.'); await loadStatus(); }}
  catch (error) {{ msg(error.message); }}
}});
document.getElementById('restart').addEventListener('click', async () => {{
  try {{ const body = await postJson('/api/restart', getOptions()); msg(`Restart lance pid=${{body.pid}}`); await loadStatus(); }}
  catch (error) {{ msg(error.message); }}
}});
loadUrls().catch(error => msg(error.message));
loadStatus().catch(error => msg(error.message));
setInterval(() => loadStatus().catch(console.error), refreshMs);
</script>
</body>
</html>
"""


def make_handler(config: AppConfig):
    class AppHandler(BaseHTTPRequestHandler):
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

        def send_json(self, data: dict[str, Any], status: int = 200) -> None:
            self.send_text(json.dumps(data, ensure_ascii=False), "application/json", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_text(dashboard_html(config.refresh_ms), "text/html")
                return
            if parsed.path == "/api/status":
                self.send_json(app_status(config))
                return
            if parsed.path == "/api/urls":
                text = config.urls_path.read_text(encoding="utf-8", errors="ignore") if config.urls_path.exists() else ""
                self.send_json({"urls_text": text})
                return
            file_map = {
                "/files/menus_lyon.json": (config.output_path, "application/json"),
                "/files/menus_lyon.csv": (config.csv_path, "text/csv"),
                "/files/nouveauurl.txt": (config.discovered_path, "text/plain"),
                "/files/scrape_menus.log": (config.log_path, "text/plain"),
            }
            if parsed.path in file_map:
                self.serve_file(*file_map[parsed.path])
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self.send_json({"ok": False, "error": "Not found"}, 404)

        def do_POST(self) -> None:
            try:
                data = read_json_body(self)
                parsed = urlparse(self.path)
                if parsed.path == "/api/urls":
                    urls_text = normalize_urls_text(str(data.get("urls_text", "")))
                    config.urls_path.write_text(urls_text, encoding="utf-8")
                    count = len([line for line in urls_text.splitlines() if line.strip()])
                    self.send_json({"ok": True, "count": count})
                    return
                if parsed.path == "/api/start":
                    result = start_scraper(config, data)
                    self.send_json(result, 200 if result.get("ok") else 409)
                    return
                if parsed.path == "/api/abort":
                    self.send_json(stop_scraper())
                    return
                if parsed.path == "/api/restart":
                    stop_scraper()
                    result = start_scraper(config, data)
                    self.send_json(result, 200 if result.get("ok") else 409)
                    return
                self.send_json({"ok": False, "error": "Not found"}, 404)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 500)

        def serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_text("File not found", status=404)
                return
            self.send_text(path.read_text(encoding="utf-8", errors="ignore"), content_type)

    return AppHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface web pour controler le scraper de menus.")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'ecoute. VPS public: 0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port HTTP.")
    parser.add_argument("--open", action="store_true", default=True, help="Ouvre le navigateur automatiquement.")
    parser.add_argument("--no-open", action="store_false", dest="open", help="N'ouvre pas le navigateur.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig(host=args.host, port=args.port)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    url = f"http://{config.host}:{config.port}/"

    print(f"Interface web: {url}")
    print("Ctrl+C pour arreter le serveur.")
    if config.host == "0.0.0.0":
        print(f"Depuis un autre poste: http://IP_DU_VPS:{config.port}/")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret du serveur.")
        stop_scraper()
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
