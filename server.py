import argparse
import json
import os
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


def _env_path(env_var: str, default: str) -> Path:
    """Lit un chemin depuis une variable d'environnement, sinon utilise le défaut."""
    return Path(os.environ.get(env_var, default))


@dataclass
class AppConfig:
    host: str
    port: int
    urls_path: Path = field(default_factory=lambda: _env_path("MENU_URLS_PATH", "urls.txt"))
    output_path: Path = field(default_factory=lambda: _env_path("MENU_OUTPUT_PATH", "menus_lyon.json"))
    csv_path: Path = field(default_factory=lambda: _env_path("MENU_CSV_PATH", "menus_lyon.csv"))
    log_path: Path = field(default_factory=lambda: _env_path("MENU_LOG_PATH", "scrape_menus.log"))
    discovered_path: Path = field(default_factory=lambda: _env_path("MENU_DISCOVERED_PATH", "nouveauurl.txt"))
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

    ai_provider = str(options.get("ai_provider", "auto") or "auto").strip()
    if ai_provider in {"openai", "gemini", "auto"}:
        command.extend(["--ai-provider", ai_provider])

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


def reset_data(config: AppConfig, what: list[str]) -> dict[str, Any]:
    """Supprime les fichiers de données selon la liste 'what'."""
    if is_running():
        return {"ok": False, "error": "Arrêtez le scraping avant de reset."}

    deleted = []
    errors = []

    def _del(path: Path, label: str) -> None:
        try:
            if path.exists():
                path.unlink()
                deleted.append(label)
        except OSError as exc:
            errors.append(f"{label}: {exc}")

    if "results" in what or "all" in what:
        _del(config.output_path, "JSON")
        csv_manquants = config.csv_path.with_name(config.csv_path.stem + "_prix_manquants" + config.csv_path.suffix)
        _del(config.csv_path, "CSV")
        _del(csv_manquants, "CSV prix manquants")
    if "logs" in what or "all" in what:
        _del(config.log_path, "Logs scraper")
        _del(Path(SERVER_STDOUT_LOG), "Logs serveur")
    if "discovered" in what or "all" in what:
        _del(config.discovered_path, "URLs découvertes")

    return {"ok": not errors, "deleted": deleted, "errors": errors}


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
<html lang="fr" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Menu Scraper</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg: #f0f2f7;
      --surface: #ffffff;
      --surface2: #f7f8fc;
      --border: #e2e8f0;
      --border2: #cbd5e1;
      --text: #0f172a;
      --text2: #475569;
      --text3: #94a3b8;
      --accent: #6366f1;
      --accent-light: #eef2ff;
      --green: #10b981;
      --green-light: #d1fae5;
      --green-dark: #065f46;
      --red: #ef4444;
      --red-light: #fee2e2;
      --red-dark: #991b1b;
      --yellow: #f59e0b;
      --yellow-light: #fef3c7;
      --yellow-dark: #78350f;
      --blue: #3b82f6;
      --blue-light: #dbeafe;
      --blue-dark: #1e3a8a;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
      --shadow: 0 4px 6px -1px rgba(0,0,0,.07), 0 2px 4px -1px rgba(0,0,0,.04);
      --radius: 12px;
      --radius-sm: 8px;
    }}

    [data-theme="dark"] {{
      --bg: #0d1117;
      --surface: #161b22;
      --surface2: #21262d;
      --border: #30363d;
      --border2: #484f58;
      --text: #e6edf3;
      --text2: #8b949e;
      --text3: #484f58;
      --accent: #818cf8;
      --accent-light: #1e1b4b;
      --green-light: #022c22;
      --green-dark: #6ee7b7;
      --red-light: #2d0e0e;
      --red-dark: #fca5a5;
      --yellow-light: #2d1a00;
      --yellow-dark: #fcd34d;
      --blue-light: #0c1a35;
      --blue-dark: #93c5fd;
    }}

    body {{
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.5;
      min-height: 100vh;
    }}

    /* === LAYOUT === */
    .app-header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 24px;
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      height: 56px;
    }}
    .app-header-left {{ display: flex; align-items: center; gap: 12px; }}
    .app-logo {{ font-size: 18px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }}
    .app-header-right {{ display: flex; align-items: center; gap: 8px; }}

    main {{ max-width: 1400px; margin: 0 auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }}

    .two-col {{ display: grid; grid-template-columns: 380px 1fr; gap: 20px; align-items: start; }}

    /* === CARDS === */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow-sm);
    }}
    .card-header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    .card-title {{ font-size: 13px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }}
    .card-body {{ padding: 20px; }}

    /* === METRICS GRID === */
    .metrics-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: var(--shadow-sm);
      transition: box-shadow 0.15s;
    }}
    .metric-card:hover {{ box-shadow: var(--shadow); }}
    .metric-label {{ font-size: 11px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: var(--text); line-height: 1; }}
    .metric-card.accent {{ border-color: var(--accent); }}
    .metric-card.accent .metric-value {{ color: var(--accent); }}
    .metric-card.green .metric-value {{ color: var(--green); }}
    .metric-card.red .metric-value {{ color: var(--red); }}
    .metric-card.yellow .metric-value {{ color: var(--yellow); }}

    /* === PROGRESS === */
    .progress-wrap {{ display: flex; flex-direction: column; gap: 6px; }}
    .progress-bar {{
      height: 8px;
      background: var(--border);
      border-radius: 999px;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--blue));
      border-radius: 999px;
      transition: width 0.4s ease;
    }}
    .progress-label {{ font-size: 12px; color: var(--text2); display: flex; justify-content: space-between; }}

    /* === STATUS BADGE === */
    .status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .status-badge.idle {{ background: var(--surface2); color: var(--text2); border: 1px solid var(--border); }}
    .status-badge.running {{ background: var(--green-light); color: var(--green-dark); border: 1px solid var(--green); }}
    .status-badge .dot {{
      width: 7px; height: 7px;
      border-radius: 50%;
      background: currentColor;
    }}
    .status-badge.running .dot {{
      animation: pulse 1.2s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.5; transform: scale(0.7); }}
    }}

    /* === ACTIVITY === */
    .activity-box {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: var(--radius-sm);
      padding: 12px 16px;
    }}
    .activity-title {{ font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
    .activity-line {{ font-size: 13px; color: var(--text2); font-family: 'JetBrains Mono', monospace; overflow-wrap: anywhere; }}

    /* === FORMS === */
    label {{ display: block; font-size: 12px; font-weight: 600; color: var(--text2); margin-bottom: 4px; }}
    input[type="text"], input[type="number"], textarea, select {{
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border2);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-size: 13px;
      color: var(--text);
      font-family: inherit;
      transition: border-color 0.15s, box-shadow 0.15s;
      outline: none;
    }}
    input:focus, textarea:focus, select:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent);
    }}
    textarea {{
      min-height: 160px;
      resize: vertical;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
    }}
    .form-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
    .form-field {{ display: flex; flex-direction: column; gap: 4px; }}

    /* === CHECKBOXES === */
    .checks {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }}
    .check-label {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      color: var(--text);
      transition: background 0.1s, border-color 0.1s;
      background: var(--surface2);
    }}
    .check-label:hover {{ background: var(--accent-light); border-color: var(--accent); }}
    .check-label input {{ width: auto; accent-color: var(--accent); }}

    /* === BUTTONS === */
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 16px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s;
      white-space: nowrap;
    }}
    .btn:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    .btn-primary {{ background: var(--accent); color: white; border-color: var(--accent); }}
    .btn-primary:hover:not(:disabled) {{ filter: brightness(1.1); box-shadow: 0 4px 12px color-mix(in srgb, var(--accent) 40%, transparent); }}
    .btn-danger {{ background: var(--red); color: white; border-color: var(--red); }}
    .btn-danger:hover:not(:disabled) {{ filter: brightness(1.1); }}
    .btn-warning {{ background: var(--yellow); color: #1c1300; border-color: var(--yellow); }}
    .btn-warning:hover:not(:disabled) {{ filter: brightness(1.05); }}
    .btn-ghost {{ background: transparent; color: var(--text2); border-color: var(--border2); }}
    .btn-ghost:hover:not(:disabled) {{ background: var(--surface2); color: var(--text); }}
    .btn-sm {{ padding: 5px 10px; font-size: 12px; }}

    .btn-group {{ display: flex; flex-wrap: wrap; gap: 8px; }}

    /* === DOWNLOAD LINKS === */
    .dl-links {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .dl-link {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 5px 10px;
      border: 1px solid var(--border2);
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 600;
      color: var(--text2);
      text-decoration: none;
      background: var(--surface2);
      transition: all 0.15s;
    }}
    .dl-link:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-light); }}

    /* === PILLS === */
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 68px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .pill-ok {{ background: var(--green-light); color: var(--green-dark); }}
    .pill-fail {{ background: var(--red-light); color: var(--red-dark); }}
    .pill-warn {{ background: var(--yellow-light); color: var(--yellow-dark); }}

    /* === TABLE === */
    .table-wrap {{ overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); box-shadow: var(--shadow-sm); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); }}
    th, td {{ padding: 10px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ background: var(--surface2); font-size: 11px; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.5px; }}
    tbody tr:hover {{ background: var(--surface2); }}
    tbody tr:last-child td {{ border-bottom: none; }}
    td a {{ color: var(--accent); text-decoration: none; }}
    td a:hover {{ text-decoration: underline; }}

    /* === LOGS === */
    .log-box {{
      background: #0d1117;
      color: #c9d1d9;
      border-radius: var(--radius);
      padding: 16px;
      max-height: 400px;
      overflow-y: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-all;
      border: 1px solid #30363d;
    }}
    .log-box .log-info {{ color: #79c0ff; }}
    .log-box .log-warn {{ color: #e3b341; }}
    .log-box .log-error {{ color: #ff7b72; }}

    /* === TOAST === */
    #toast-container {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 9999;
    }}
    .toast {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      border-radius: var(--radius-sm);
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      font-size: 13px;
      font-weight: 500;
      min-width: 240px;
      max-width: 380px;
      animation: slideIn 0.2s ease;
    }}
    .toast.success {{ border-left: 3px solid var(--green); }}
    .toast.error {{ border-left: 3px solid var(--red); }}
    .toast.info {{ border-left: 3px solid var(--accent); }}
    @keyframes slideIn {{ from {{ opacity: 0; transform: translateX(20px); }} to {{ opacity: 1; transform: translateX(0); }} }}
    @keyframes slideOut {{ from {{ opacity: 1; }} to {{ opacity: 0; transform: translateX(20px); }} }}

    /* === DIVIDER === */
    .divider {{ height: 1px; background: var(--border); margin: 16px 0; }}

    /* === DARK TOGGLE === */
    .theme-toggle {{
      width: 36px; height: 36px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border2);
      background: var(--surface2);
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
      color: var(--text2);
      transition: all 0.15s;
    }}
    .theme-toggle:hover {{ border-color: var(--accent); color: var(--accent); }}

    /* === RESPONSIVE === */
    @media (max-width: 1100px) {{
      .two-col {{ grid-template-columns: 1fr; }}
      .metrics-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 640px) {{
      main {{ padding: 16px; }}
      .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .app-header {{ padding: 0 16px; }}
    }}

    /* === SEPARATOR === */
    .section-gap {{ display: flex; flex-direction: column; gap: 16px; }}
  </style>
</head>
<body>

<header class="app-header">
  <div class="app-header-left">
    <span class="app-logo">&#127860; MenuScraper</span>
    <span id="job-badge" class="status-badge idle"><span class="dot"></span>Idle</span>
  </div>
  <div class="app-header-right">
    <div class="dl-links">
      <a class="dl-link" href="/files/menus_lyon.json" target="_blank">&#8595; JSON</a>
      <a class="dl-link" href="/files/menus_lyon.csv" target="_blank">&#8595; CSV</a>
      <a class="dl-link" href="/files/nouveauurl.txt" target="_blank">&#8595; URLs</a>
      <a class="dl-link" href="/files/scrape_menus.log" target="_blank">&#128196; Logs</a>
    </div>
    <button class="theme-toggle" id="themeToggle" title="Basculer le theme">&#9790;</button>
  </div>
</header>

<main>
  <!-- METRICS -->
  <section class="metrics-grid">
    <div class="metric-card accent">
      <div class="metric-label">Progression</div>
      <div class="metric-value" id="processed">0/0</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Restants</div>
      <div class="metric-value" id="remaining">0</div>
    </div>
    <div class="metric-card green">
      <div class="metric-label">Menus OK</div>
      <div class="metric-value" id="success">0</div>
    </div>
    <div class="metric-card red">
      <div class="metric-label">A revoir</div>
      <div class="metric-value" id="failed">0</div>
    </div>
    <div class="metric-card yellow">
      <div class="metric-label">Prix incomplets</div>
      <div class="metric-value" id="incomplete">0</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Plats exportes</div>
      <div class="metric-value" id="dishes">0</div>
    </div>
  </section>

  <!-- PROGRESS BAR -->
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="progressbar"></div></div>
    <div class="progress-label">
      <span id="progress-pct">0%</span>
      <span id="updated" class=""></span>
    </div>
  </div>

  <div class="two-col">
    <!-- LEFT: CONTROL PANEL -->
    <div class="section-gap">
      <!-- URL INPUT -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">&#127759; URLs a scraper</span>
          <label class="btn btn-ghost btn-sm" style="cursor:pointer; margin:0;">
            &#128194; Importer
            <input id="urlFile" type="file" accept=".txt,text/plain" style="display:none">
          </label>
        </div>
        <div class="card-body section-gap">
          <div class="form-field">
            <label for="urlsText">Une URL par ligne</label>
            <textarea id="urlsText" placeholder="https://example-restaurant.com&#10;https://another-place.fr"></textarea>
          </div>
          <div class="btn-group">
            <button id="saveUrls" class="btn btn-ghost">&#128190; Enregistrer</button>
            <button id="reloadUrls" class="btn btn-ghost">&#8635; Recharger</button>
          </div>
        </div>
      </div>

      <!-- OPTIONS -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">&#9881; Options</span>
          <button id="resetOptions" class="btn btn-ghost btn-sm">&#8635; Reset</button>
        </div>
        <div class="card-body section-gap">
          <div class="form-grid">
            <div class="form-field">
              <label>&#129302; IA</label>
              <select id="ai_provider" style="width:100%;padding:8px 10px;border:1px solid var(--border2);border-radius:8px;background:var(--surface);color:var(--text);font-size:14px">
                <option value="auto">Auto (OpenAI → Gemini)</option>
                <option value="gemini">Gemini uniquement</option>
                <option value="openai">OpenAI uniquement</option>
              </select>
            </div>
            <div class="form-field"><label>Workers</label><input id="workers" type="number" min="1" max="12" value="3"></div>
            <div class="form-field"><label>Pages menu max</label><input id="max_menu_pages" type="number" min="1" max="10" value="4"></div>
            <div class="form-field"><label>Zyte retries</label><input id="zyte_retries" type="number" min="0" max="10" value="3"></div>
            <div class="form-field"><label>Zyte timeout (s)</label><input id="zyte_timeout" type="number" min="15" max="180" value="60"></div>
            <div class="form-field"><label>OpenAI timeout (s)</label><input id="openai_timeout" type="number" min="15" max="180" value="90"></div>
            <div class="form-field"><label>Pause inter-jobs (s)</label><input id="sleep" type="number" min="0" max="10" step="0.1" value="0.2"></div>
            <div class="form-field"><label>Zyte IP Type</label>
              <select id="zyte_ip_type">
                <option value="">Auto</option>
                <option value="datacenter">Datacenter</option>
                <option value="residential">Residential</option>
              </select>
            </div>
            <div class="form-field"><label>Geolocalisation</label><input id="zyte_geolocation" value="FR" type="text"></div>
          </div>
          <div class="checks">
            <label class="check-label"><input id="retry_failed" type="checkbox" checked> Retry failed</label>
            <label class="check-label"><input id="retry_incomplete" type="checkbox" checked> Retry incomplets</label>
            <label class="check-label"><input id="overwrite" type="checkbox"> Overwrite</label>
            <label class="check-label"><input id="keep_duplicates" type="checkbox"> Garder doublons</label>
            <label class="check-label"><input id="no_interactive_expand" type="checkbox"> Sans clic onglets</label>
            <label class="check-label"><input id="no_network_capture" type="checkbox"> Sans capture reseau</label>
          </div>
        </div>
      </div>

      <!-- ACTIONS -->
      <div class="card">
        <div class="card-body">
          <div class="btn-group">
            <button id="start" class="btn btn-primary">&#9654; Lancer</button>
            <button id="abort" class="btn btn-danger" disabled>&#9632; Arreter</button>
            <button id="restart" class="btn btn-warning">&#8635; Restart</button>
            <button id="resetAll" class="btn" style="background:var(--surface2);color:var(--text2);border:1px solid var(--border2)" title="Remet a zero les resultats, logs et stats">&#128465; Reset tout</button>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT: MONITORING -->
    <div class="section-gap">
      <!-- ACTIVITY -->
      <div class="activity-box">
        <div class="activity-title" id="current-title">Activite</div>
        <div class="activity-line" id="current-line">Chargement...</div>
      </div>

      <!-- RESULTS TABLE -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">&#128203; Derniers resultats</span>
          <span id="results-count" class="pill pill-ok" style="display:none"></span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Restaurant</th>
                <th>Statut</th>
                <th>Plats</th>
                <th>Categories</th>
                <th>Erreur</th>
              </tr>
            </thead>
            <tbody id="results"><tr><td colspan="5" style="color:var(--text3);text-align:center;padding:24px">Aucun resultat pour le moment.</td></tr></tbody>
          </table>
        </div>
      </div>

      <!-- LOGS -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">&#128196; Logs</span>
          <button id="clearLogs" class="btn btn-ghost btn-sm">Effacer affichage</button>
        </div>
        <div class="card-body" style="padding:12px">
          <div class="log-box" id="logs">Chargement...</div>
        </div>
      </div>
    </div>
  </div>
</main>

<div id="toast-container"></div>

<script>
const refreshMs = {refresh_ms};
const FIELDS = ['workers','max_menu_pages','zyte_retries','zyte_timeout','openai_timeout','sleep','zyte_ip_type','zyte_geolocation'];
const SELECTS = ['ai_provider'];
const CHECKS = ['retry_failed','retry_incomplete','overwrite','keep_duplicates','no_interactive_expand','no_network_capture'];
const DEFAULTS = {{ workers:'3', max_menu_pages:'4', zyte_retries:'3', zyte_timeout:'60', openai_timeout:'90', sleep:'0.2', zyte_ip_type:'', zyte_geolocation:'FR' }};
const DEFAULTS_SELECTS = {{ ai_provider:'auto' }};
const DEFAULTS_CHECKS = {{ retry_failed:true, retry_incomplete:true, overwrite:false, keep_duplicates:false, no_interactive_expand:false, no_network_capture:false }};

// === THEME ===
function applyTheme(t) {{
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('themeToggle').textContent = t === 'dark' ? '☀' : '☽';
  localStorage.setItem('theme', t);
}}
document.getElementById('themeToggle').addEventListener('click', () => {{
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
}});
applyTheme(localStorage.getItem('theme') || 'light');

// === TOAST ===
function toast(message, type = 'info') {{
  const el = document.createElement('div');
  el.className = `toast ${{type}}`;
  el.textContent = message;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => {{
    el.style.animation = 'slideOut 0.2s ease forwards';
    setTimeout(() => el.remove(), 200);
  }}, 3500);
}}

// === OPTIONS PERSISTENCE ===
function saveOptions() {{
  const data = {{}};
  for (const id of FIELDS) data[id] = document.getElementById(id).value;
  for (const id of SELECTS) data[id] = document.getElementById(id).value;
  for (const id of CHECKS) data[id] = document.getElementById(id).checked;
  localStorage.setItem('scraper_options', JSON.stringify(data));
}}
function loadOptions() {{
  const raw = localStorage.getItem('scraper_options');
  if (!raw) return;
  try {{
    const data = JSON.parse(raw);
    for (const id of FIELDS) if (data[id] !== undefined) document.getElementById(id).value = data[id];
    for (const id of SELECTS) if (data[id] !== undefined) document.getElementById(id).value = data[id];
    for (const id of CHECKS) if (data[id] !== undefined) document.getElementById(id).checked = data[id];
  }} catch(e) {{}}
}}
function resetOptions() {{
  for (const [id, val] of Object.entries(DEFAULTS)) document.getElementById(id).value = val;
  for (const [id, val] of Object.entries(DEFAULTS_SELECTS)) document.getElementById(id).value = val;
  for (const [id, val] of Object.entries(DEFAULTS_CHECKS)) document.getElementById(id).checked = val;
  localStorage.removeItem('scraper_options');
  toast('Options remises par defaut', 'info');
}}
loadOptions();
[...FIELDS, ...SELECTS, ...CHECKS].forEach(id => {{
  document.getElementById(id).addEventListener('change', saveOptions);
}});
document.getElementById('resetOptions').addEventListener('click', resetOptions);

// === HELPERS ===
function setText(id, value) {{ document.getElementById(id).textContent = value; }}
function getOptions() {{
  const data = {{}};
  for (const id of FIELDS) data[id] = document.getElementById(id).value;
  for (const id of SELECTS) data[id] = document.getElementById(id).value;
  for (const id of CHECKS) data[id] = document.getElementById(id).checked;
  return data;
}}
async function postJson(url, data) {{
  const r = await fetch(url, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(data || {{}})
  }});
  const body = await r.json();
  if (!r.ok || body.ok === false) throw new Error(body.error || body.message || 'Erreur serveur');
  return body;
}}

function pill(item) {{
  if (item.incomplete) return '<span class="pill pill-warn">Prix</span>';
  if (item.menu) return '<span class="pill pill-ok">OK</span>';
  return '<span class="pill pill-fail">A revoir</span>';
}}

function colorLog(line) {{
  const l = line.toLowerCase();
  if (l.includes('| error |') || l.includes('erreur') || l.includes('echec')) return `<span class="log-error">${{escHtml(line)}}</span>`;
  if (l.includes('| warning |') || l.includes('warn') || l.includes('attention')) return `<span class="log-warn">${{escHtml(line)}}</span>`;
  return `<span class="log-info">${{escHtml(line)}}</span>`;
}}
function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

// === URLS ===
async function loadUrls() {{
  const res = await fetch('/api/urls?ts=' + Date.now());
  const data = await res.json();
  document.getElementById('urlsText').value = data.urls_text || '';
}}

// === STATUS ===
let lastLogLine = '';

async function loadStatus() {{
  const r = await fetch('/api/status?ts=' + Date.now());
  if (!r.ok) throw new Error('Erreur serveur ' + r.status);
  const data = await r.json();

  setText('updated', 'Mis a jour: ' + data.generated_at);
  setText('processed', (data.processed || 0) + '/' + (data.total || 0));
  setText('remaining', data.remaining || 0);
  setText('success', data.success || 0);
  setText('failed', data.failed || 0);
  setText('incomplete', data.incomplete || 0);
  setText('dishes', data.dishes || 0);

  const pct = data.progress || 0;
  document.getElementById('progressbar').style.width = pct + '%';
  setText('progress-pct', Math.round(pct) + '%');

  const running = !!(data.job && data.job.running);
  const badge = document.getElementById('job-badge');
  badge.className = running ? 'status-badge running' : 'status-badge idle';
  badge.innerHTML = running ? '<span class="dot"></span>Running' : '<span class="dot"></span>Idle';
  document.getElementById('start').disabled = running;
  document.getElementById('abort').disabled = !running;

  const ct = data.current || {{}};
  const actTitle = (ct.index || ct.domain) ? `Activite ${{ct.index || ''}} ${{ct.domain || ''}}` : 'Activite';
  setText('current-title', actTitle);
  setText('current-line', ct.line || '—');

  const items = (data.recent_results || []).slice().reverse();
  const tbody = document.getElementById('results');
  if (items.length === 0) {{
    tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text3);text-align:center;padding:24px">Aucun resultat pour le moment.</td></tr>';
  }} else {{
    tbody.innerHTML = items.map(item => `
      <tr>
        <td><a href="${{escHtml(item.url)}}" target="_blank">${{escHtml(item.domain || item.url)}}</a></td>
        <td>${{pill(item)}}</td>
        <td>${{item.dishes}}</td>
        <td>${{item.categories}}</td>
        <td style="color:var(--text3);font-size:12px">${{escHtml(item.error || '')}}</td>
      </tr>
    `).join('');
    const rc = document.getElementById('results-count');
    rc.style.display = '';
    rc.textContent = items.length + ' resultats';
  }}

  const logs = data.logs || [];
  const newLastLine = logs.length ? logs[logs.length - 1] : '';
  if (newLastLine !== lastLogLine) {{
    lastLogLine = newLastLine;
    const logEl = document.getElementById('logs');
    logEl.innerHTML = logs.map(colorLog).join('\\n');
    logEl.scrollTop = logEl.scrollHeight;
  }}
}}

// === EVENTS ===
document.getElementById('urlFile').addEventListener('change', async (e) => {{
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('urlsText').value = await file.text();
  toast('Fichier charge: ' + file.name, 'info');
}});

document.getElementById('saveUrls').addEventListener('click', async () => {{
  try {{
    const body = await postJson('/api/urls', {{ urls_text: document.getElementById('urlsText').value }});
    toast(`${{body.count}} URL(s) enregistree(s)`, 'success');
    await loadStatus();
  }} catch(e) {{ toast(e.message, 'error'); }}
}});

document.getElementById('reloadUrls').addEventListener('click', () =>
  loadUrls().then(() => toast('URLs rechargees', 'info')).catch(e => toast(e.message, 'error'))
);

document.getElementById('start').addEventListener('click', async () => {{
  try {{
    const body = await postJson('/api/start', getOptions());
    toast(`Scraping lance (pid ${{body.pid}})`, 'success');
    await loadStatus();
  }} catch(e) {{ toast(e.message, 'error'); }}
}});

document.getElementById('abort').addEventListener('click', async () => {{
  try {{
    const body = await postJson('/api/abort', {{}});
    toast(body.message || 'Arrete.', 'info');
    await loadStatus();
  }} catch(e) {{ toast(e.message, 'error'); }}
}});

document.getElementById('restart').addEventListener('click', async () => {{
  try {{
    const body = await postJson('/api/restart', getOptions());
    toast(`Restart lance (pid ${{body.pid}})`, 'success');
    await loadStatus();
  }} catch(e) {{ toast(e.message, 'error'); }}
}});

document.getElementById('clearLogs').addEventListener('click', () => {{
  document.getElementById('logs').innerHTML = '';
  lastLogLine = '';
}});

document.getElementById('resetAll').addEventListener('click', async () => {{
  if (!confirm('Supprimer tous les resultats, CSV, logs et stats ? Cette action est irreversible.')) return;
  try {{
    const body = await postJson('/api/reset', {{ what: ['all'] }});
    if (body.ok) {{
      toast('Reset effectue: ' + (body.deleted || []).join(', '), 'success');
    }} else {{
      toast('Reset partiel: ' + (body.errors || []).join(', '), 'error');
    }}
    lastLogLine = '';
    document.getElementById('logs').innerHTML = '';
    await loadStatus();
  }} catch(e) {{ toast(e.message, 'error'); }}
}});

// === INIT ===
loadUrls().catch(console.error);
loadStatus().catch(console.error);
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
                if parsed.path == "/api/reset":
                    what = data.get("what", ["all"])
                    self.send_json(reset_data(config, what))
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
