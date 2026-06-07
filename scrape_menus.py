import argparse
import base64
import csv
import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types as google_genai_types
from openai import APIError, APITimeoutError, OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ZYTE_ENDPOINT = "https://api.zyte.com/v1/extract"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
PRICE_PATTERN = re.compile(
    "(?<!\\d)(\\d{1,3}(?:[,.]\\d{1,2})?)\\s*(?:\\u20ac|eur|euros?|euro|EUR|EURS?)\\.?(?![a-zA-Z])|"
    "\\u20ac\\s*(\\d{1,3}(?:[,.]\\d{1,2})?)",
    re.IGNORECASE,
)
MENU_LINK_KEYWORDS = {
    "menu",
    "menus",
    "carte",
    "la-carte",
    "notre-carte",
    "notre carte",
    "services",
    "service",
    "plats",
    "dish",
    "dishes",
    "carta",
    "comida",
    "platos",
    "entrantes",
    "postres",
    "bebidas",
    "pizzas",
    "burgers",
    "tacos",
    "commander",
    "commande",
    "order",
    "food",
    "restaurant",
}
MENU_TEXT_KEYWORDS = {
    "menu",
    "carte",
    "entree",
    "entrees",
    "plat",
    "plats",
    "dessert",
    "desserts",
    "pizza",
    "pizzas",
    "burger",
    "burgers",
    "tacos",
    "salade",
    "salades",
    "boisson",
    "boissons",
    "vin",
    "formule",
    "supplement",
    "starter",
    "starters",
    "main",
    "dish",
    "dishes",
    "drink",
    "drinks",
    "carta",
    "comida",
    "entrante",
    "entrantes",
    "plato",
    "platos",
    "postre",
    "postres",
    "bebida",
    "bebidas",
}
EXCLUDED_LINK_KEYWORDS = {
    "facebook",
    "instagram",
    "tiktok",
    "tripadvisor",
    "maps",
    "mailto:",
    "tel:",
    "reservation",
    "reserver",
    "contact",
    "avis",
    "privacy",
    "confidentialite",
    "mentions",
    "conditions",
    "panier",
    "checkout",
    "login",
    "connexion",
}
UNSUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
}


@dataclass
class FetchedPage:
    html: str
    final_url: str
    network_text: str = ""
    action_errors: list[str] = field(default_factory=list)


@dataclass
class MenuLinkCandidate:
    url: str
    text: str
    score: int
    is_media: bool = False  # True si PDF ou image (OCR nécessaire)


@dataclass
class ProcessOutcome:
    result: dict
    discovered_urls: list[str] = field(default_factory=list)


class UnsupportedContentError(Exception):
    """Raised when the URL is a PDF, image, or another unsupported resource."""


class DeadPageError(Exception):
    """Raised when Zyte reaches the page but the target response is not usable."""


class ZyteDownloadError(Exception):
    """Raised when Zyte cannot provide a response for the requested URL."""

    def __init__(self, status_code: int, detail: str, retryable: bool) -> None:
        super().__init__(f"Erreur Zyte HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.retryable = retryable


class Plat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nom_plat: str = Field(description="Nom exact du plat tel qu'il apparait.")
    description_ingredients: Optional[str] = Field(
        default=None,
        description="Ingredients ou courte description. Null si non precise.",
    )
    prix_euros: Optional[float] = Field(
        default=None,
        description="Prix en euros. Null si le prix est absent ou impossible a verifier.",
    )


class Categorie(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nom_categorie: str = Field(description="Nom de categorie: Entrees, Pizzas, Desserts...")
    plats: list[Plat] = Field(default_factory=list)


class RestaurantMenu(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url_restaurant: str
    menu_disponible: bool
    categories: list[Categorie] = Field(default_factory=list)


class FlatMenuItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    categorie_plat: str = Field(description="Categorie du plat: Entrees, Burgers, Desserts...")
    nom_plat: str = Field(description="Nom exact du plat.")
    description: Optional[str] = Field(
        default=None,
        description="Ingredients ou description. Null si non precise.",
    )
    prix: Optional[float] = Field(
        default=None,
        description="Prix en euros, uniquement le nombre. Null si absent ou ambigu.",
    )


class AIFlatMenuExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    menu_disponible: bool
    erreur: Optional[str] = Field(
        default=None,
        description="Erreur explicite si aucun menu exploitable, lien mort, PDF/image, ou menu absent.",
    )
    items: list[FlatMenuItem] = Field(default_factory=list)


def configure_logging(log_file: Optional[str] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def normalize_input_url(raw_url: str) -> str:
    url = raw_url.strip().strip('"').strip("'")
    url = url.replace("\\_", "_").replace("\\&", "&")
    url = url.replace("&amp;", "&")
    if url.startswith("www."):
        url = "https://" + url
    return url


def read_urls(path: Path, keep_duplicates: bool = False) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier d'URLs introuvable: {path}")

    urls: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = normalize_input_url(raw_line)
        if not line or line.startswith("#"):
            continue
        if not keep_duplicates and line in seen:
            duplicate_count += 1
            continue
        seen.add(line)
        urls.append(line)

    if duplicate_count:
        logging.info("%s URL(s) exacte(s) en doublon ignoree(s)", duplicate_count)

    return urls


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Le fichier de sortie existe mais n'est pas un JSON valide: %s", path)
        return []

    if not isinstance(data, list):
        logging.warning("Le fichier de sortie doit contenir une liste JSON: %s", path)
        return []

    return data


def save_results_atomic(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def upsert_result(results: list[dict], result: dict) -> None:
    url = result.get("url_restaurant")
    for index, existing in enumerate(results):
        if isinstance(existing, dict) and existing.get("url_restaurant") == url:
            results[index] = result
            return
    results.append(result)


def sort_results_by_input_order(results: list[dict], urls: list[str]) -> None:
    order = {url: index for index, url in enumerate(urls)}
    results.sort(
        key=lambda item: (
            order.get(item.get("url_restaurant"), len(order)),
            item.get("url_restaurant", ""),
        )
    )


def load_discovered_urls(path: Path) -> list[str]:
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


def save_discovered_urls_atomic(path: Path, new_urls: list[str]) -> None:
    if not path:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    urls = load_discovered_urls(path)
    seen = set(urls)

    for raw_url in new_urls:
        url = normalize_input_url(raw_url)
        if not url.startswith(("http://", "https://")):
            continue
        if looks_like_unsupported_resource(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    tmp_path.replace(path)


def empty_result(url: str) -> dict:
    return {
        "url_restaurant": url,
        "menu_disponible": False,
        "categories": [],
    }


def error_result(url: str, error_message: str) -> dict:
    result = empty_result(url)
    result["erreur"] = error_message
    return result


def looks_like_unsupported_resource(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in UNSUPPORTED_EXTENSIONS)


def domain_for_log(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url[:80]


def zyte_headers_to_dict(headers: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for header in headers or []:
        name = str(header.get("name", "")).lower()
        value = str(header.get("value", ""))
        if name:
            result[name] = value
    return result


def build_browser_actions(expand_interactive: bool) -> list[dict]:
    actions: list[dict] = [
        {"action": "scrollBottom", "onError": "continue"},
        {"action": "waitForTimeout", "timeout": 1, "onError": "continue"},
    ]

    if expand_interactive:
        actions.extend(
            [
                {
                    "action": "evaluate",
                    "source": """
const keywords = [
  'menu', 'carte', 'entree', 'entrees', 'plat', 'plats', 'dessert', 'desserts',
  'pizza', 'pizzas', 'burger', 'burgers', 'tacos', 'boisson', 'boissons'
];
const nodes = Array.from(document.querySelectorAll(
  'button,[role="tab"],[role="button"],[aria-controls],.tab,.tabs li,.accordion button,a[role="tab"],a[role="button"]'
));
let clicked = 0;
for (const node of nodes) {
  const text = (
    node.innerText || node.textContent || node.getAttribute('aria-label') || ''
  ).toLowerCase();
  if (!keywords.some((keyword) => text.includes(keyword))) continue;
  if (node.tagName === 'A') {
    const href = node.getAttribute('href') || '';
    if (href && !href.startsWith('#') && !href.startsWith('javascript:')) continue;
  }
  try {
    node.click();
    clicked += 1;
  } catch (_) {}
  if (clicked >= 20) break;
}
console.log(`menu_expand_clicks=${clicked}`);
""".strip(),
                    "onError": "continue",
                },
                {"action": "waitForTimeout", "timeout": 2, "onError": "continue"},
                {"action": "scrollBottom", "onError": "continue"},
                {"action": "waitForTimeout", "timeout": 1, "onError": "continue"},
            ]
        )

    return actions


def build_network_capture_filters() -> list[dict]:
    return [
        {"filterType": "url", "value": "menu", "matchType": "contains", "httpResponseBody": True},
        {"filterType": "url", "value": "carte", "matchType": "contains", "httpResponseBody": True},
        {"filterType": "url", "value": "api", "matchType": "contains", "httpResponseBody": True},
        {"filterType": "url", "value": "catalog", "matchType": "contains", "httpResponseBody": True},
        {"filterType": "url", "value": "product", "matchType": "contains", "httpResponseBody": True},
    ]


def decode_network_capture(data: dict, max_chars: int = 18000) -> str:
    chunks: list[str] = []
    for item in data.get("networkCapture", []) or []:
        body = item.get("httpResponseBody")
        if not body:
            continue
        try:
            decoded = base64.b64decode(body).decode("utf-8", errors="ignore")
        except (ValueError, TypeError):
            continue

        compact = re.sub(r"\s+", " ", decoded).strip()
        if not compact or len(compact) < 40:
            continue
        if not (PRICE_PATTERN.search(compact) or any(keyword in compact.casefold() for keyword in MENU_TEXT_KEYWORDS)):
            continue

        source_url = item.get("url", "network")
        chunks.append(f"NETWORK_SOURCE: {source_url}\n{compact[:max_chars]}")

    return "\n\n".join(chunks)[:max_chars]


def extract_action_errors(data: dict) -> list[str]:
    errors: list[str] = []
    for action in data.get("actions", []) or []:
        if action.get("status") in {"continued", "returned"} and action.get("error"):
            errors.append(str(action["error"])[:250])
    return errors


def build_zyte_payload(
    url: str,
    ip_type: Optional[str],
    geolocation: Optional[str],
    expand_interactive: bool,
    capture_network: bool,
    include_iframes: bool,
) -> dict:
    payload = {
        "url": url,
        "browserHtml": True,
        "httpResponseHeaders": True,
        "viewport": {"width": 1366, "height": 900},
        "device": "desktop",
    }

    if include_iframes:
        payload["includeIframes"] = True
    if expand_interactive:
        payload["actions"] = build_browser_actions(expand_interactive)
    if capture_network:
        payload["networkCapture"] = build_network_capture_filters()
    if ip_type:
        payload["ipType"] = ip_type
    if geolocation:
        payload["geolocation"] = geolocation.upper()

    return payload


def zyte_payload_mode(payload: dict) -> str:
    if payload.get("actions") or payload.get("networkCapture"):
        return "enriched"
    if payload.get("includeIframes"):
        return "lite"
    return "lite-no-iframes"


def downgrade_zyte_payload(payload: dict, url: str, reason: str) -> bool:
    if payload.pop("networkCapture", None) is not None or payload.pop("actions", None) is not None:
        logging.warning(
            "Zyte fallback pour %s: mode enrichi trop lourd (%s), retry en mode lite",
            domain_for_log(url),
            reason,
        )
        return True

    if payload.pop("includeIframes", None) is not None:
        logging.warning(
            "Zyte fallback pour %s: iframes trop lourds (%s), retry en mode lite-no-iframes",
            domain_for_log(url),
            reason,
        )
        return True

    return False


def fetch_html_with_zyte(
    session: requests.Session,
    url: str,
    zyte_api_key: str,
    timeout: int,
    retries: int,
    ip_type: Optional[str],
    geolocation: Optional[str],
    expand_interactive: bool,
    capture_network: bool,
) -> FetchedPage:
    payload = build_zyte_payload(
        url=url,
        ip_type=ip_type,
        geolocation=geolocation,
        expand_interactive=expand_interactive,
        capture_network=capture_network,
        include_iframes=True,
    )
    last_error: Optional[ZyteDownloadError] = None

    for attempt in range(1, retries + 2):
        mode = zyte_payload_mode(payload)
        logging.info(
            "Zyte tentative %s/%s pour %s en mode %s",
            attempt,
            retries + 1,
            domain_for_log(url),
            mode,
        )

        try:
            response = session.post(
                ZYTE_ENDPOINT,
                auth=(zyte_api_key, ""),
                json=payload,
                timeout=timeout,
                headers={"Accept-Encoding": "gzip"},
            )
        except requests.Timeout:
            if downgrade_zyte_payload(payload, url, "timeout"):
                continue
            if attempt > retries:
                raise
            wait_for_retry(attempt, "Timeout Zyte", url=url, mode=mode)
            continue

        if response.status_code >= 400:
            detail = response.text[:500].replace("\n", " ")
            if response.status_code in {400, 520, 521} and downgrade_zyte_payload(
                payload,
                url,
                f"HTTP {response.status_code}",
            ):
                continue

            retryable = response.status_code in {429, 500, 503, 520, 521}
            last_error = ZyteDownloadError(response.status_code, detail, retryable)

            if retryable and attempt <= retries:
                wait_for_retry(attempt, str(last_error), url=url, mode=mode)
                continue

            raise last_error

        break
    else:
        if last_error:
            raise last_error
        raise DeadPageError("Zyte n'a pas retourne de reponse exploitable")

    data = response.json()
    final_url = data.get("url") or url
    status_code = data.get("statusCode")
    headers = zyte_headers_to_dict(data.get("httpResponseHeaders", []))
    content_type = headers.get("content-type", "").lower()

    if status_code and int(status_code) >= 400:
        raise DeadPageError(f"Page cible en erreur HTTP {status_code}")

    if "application/pdf" in content_type or content_type.startswith("image/"):
        raise UnsupportedContentError(f"Ressource non HTML detectee: {content_type}")

    html = data.get("browserHtml")
    if not html or not isinstance(html, str):
        raise UnsupportedContentError("Zyte n'a retourne aucun HTML exploitable")

    return FetchedPage(
        html=html,
        final_url=final_url,
        network_text=decode_network_capture(data),
        action_errors=extract_action_errors(data),
    )


def wait_for_retry(attempt: int, reason: str, url: Optional[str] = None, mode: Optional[str] = None) -> None:
    upper_bound = min(62, 6 * (2 ** (attempt - 1)))
    sleep_seconds = random.uniform(3, upper_bound)
    target = f" pour {domain_for_log(url)}" if url else ""
    mode_label = f" en mode {mode}" if mode else ""
    logging.warning(
        "Zyte retry %s%s%s apres erreur: %s | attente %.1fs",
        attempt,
        target,
        mode_label,
        reason,
        sleep_seconds,
    )
    time.sleep(sleep_seconds)


def remove_noise_nodes(soup: BeautifulSoup) -> None:
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "canvas",
            "footer",
            "aside",
        ]
    ):
        tag.decompose()

    noise_pattern = re.compile(
        r"(navbar|navigation|cookie|consent|gdpr|popup|modal|newsletter|"
        r"breadcrumb|sidebar|social|share|legal|tracking|analytics)",
        re.IGNORECASE,
    )

    for tag in soup.find_all(True):
        attrs_dict = tag.attrs or {}
        attrs = " ".join(
            str(value)
            for key, value in attrs_dict.items()
            if key in {"class", "id", "role", "aria-label"}
        )
        if attrs and noise_pattern.search(attrs):
            text_preview = tag.get_text(" ", strip=True)[:5000]
            if PRICE_PATTERN.search(text_preview) or any(
                keyword in text_preview.casefold() for keyword in MENU_TEXT_KEYWORDS
            ):
                continue
            tag.decompose()


def is_boilerplate_line(line: str) -> bool:
    normalized = line.casefold()
    boilerplate_fragments = {
        "mentions legales",
        "politique de confidentialite",
        "gestion des cookies",
        "accepter les cookies",
        "refuser les cookies",
        "tous droits reserves",
        "powered by",
        "facebook",
        "instagram",
        "tripadvisor",
        "google maps",
        "copyright",
    }
    return any(fragment in normalized for fragment in boilerplate_fragments)


def clean_html_to_visible_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    remove_noise_nodes(soup)

    raw_text = soup.get_text("\n")
    lines: list[str] = []
    seen: set[str] = set()

    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if len(line) <= 2 and not re.search(r"\d", line):
            continue
        if is_boilerplate_line(line):
            continue

        dedupe_key = line.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        lines.append(line)

    return trim_text_for_llm(lines, max_chars=max_chars)


def trim_text_for_llm(lines: list[str], max_chars: int) -> str:
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    interesting_indexes: set[int] = set()
    for index, line in enumerate(lines):
        lowered = line.casefold()
        if PRICE_PATTERN.search(line) or any(keyword in lowered for keyword in MENU_TEXT_KEYWORDS):
            for neighbor in range(max(0, index - 3), min(len(lines), index + 4)):
                interesting_indexes.add(neighbor)

    if not interesting_indexes:
        return full_text[:max_chars]

    selected = [lines[index] for index in sorted(interesting_indexes)]
    prioritized = "\n".join(selected)

    if len(prioritized) < max_chars * 0.65:
        prefix = "\n".join(lines[:80])
        prioritized = prefix + "\n...\n" + prioritized

    return prioritized[:max_chars]


def normalize_for_match(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def score_menu_link(text: str, href: str, base_url: str) -> int:
    haystack = normalize_for_match(f"{text} {href}")
    parsed_base = urlparse(base_url)
    parsed_href = urlparse(href)

    score = 0
    if parsed_href.netloc == parsed_base.netloc:
        score += 2

    for keyword in MENU_LINK_KEYWORDS:
        normalized_keyword = normalize_for_match(keyword)
        if normalized_keyword and normalized_keyword in haystack:
            score += 5 if normalized_keyword in {"menu", "carte", "la carte", "notre carte"} else 2

    for keyword in EXCLUDED_LINK_KEYWORDS:
        if normalize_for_match(keyword) in haystack:
            score -= 8

    path = parsed_href.path.casefold()
    if path in {"", "/"}:
        score -= 3
    # Ne pénalise plus les PDFs/images : on les gère via OCR
    if href.rstrip("/") == base_url.rstrip("/"):
        score -= 5

    return score


def discover_menu_links(html: str, base_url: str, limit: int) -> list[MenuLinkCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates_by_url: dict[str, MenuLinkCandidate] = {}

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue

        absolute_url = normalize_input_url(urljoin(base_url, href))
        absolute_url = urldefrag(absolute_url)[0]
        if not absolute_url.startswith(("http://", "https://")):
            continue

        text = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True))
        score = score_menu_link(text, absolute_url, base_url)
        is_media = looks_like_unsupported_resource(absolute_url)

        # Les PDF/images doivent avoir un score plus fort pour être inclus
        min_score = 6 if is_media else 4
        if score < min_score:
            continue

        existing = candidates_by_url.get(absolute_url)
        if existing is None or score > existing.score:
            candidates_by_url[absolute_url] = MenuLinkCandidate(
                url=absolute_url,
                text=text[:120],
                score=score,
                is_media=is_media,
            )

    return sorted(candidates_by_url.values(), key=lambda item: item.score, reverse=True)[:limit]


def menu_density_score(text: str) -> int:
    price_count = len(PRICE_PATTERN.findall(text))
    lowered = normalize_for_match(text[:60000])
    keyword_count = sum(lowered.count(normalize_for_match(keyword)) for keyword in MENU_TEXT_KEYWORDS)
    return price_count * 4 + keyword_count


def combine_page_texts(page_texts: list[tuple[str, str]], max_chars: int) -> str:
    chunks = [
        f"SOURCE_PAGE: {source_url}\n{text.strip()}"
        for source_url, text in page_texts
        if text and text.strip()
    ]
    combined = "\n\n---\n\n".join(chunks)
    if len(combined) <= max_chars:
        return combined

    return trim_text_for_llm(combined.splitlines(), max_chars=max_chars)


def _extract_pdf_text_pdfplumber(content: bytes) -> str:
    """Extrait le texte d'un PDF avec pdfplumber. Retourne '' si pas de texte (PDF scanné)."""
    try:
        import io
        import pdfplumber  # type: ignore[import]

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            parts: list[str] = []
            for page in pdf.pages[:10]:
                text = page.extract_text()
                if text:
                    parts.append(text)
            return "\n\n".join(parts)
    except Exception as exc:
        logging.debug("pdfplumber echec: %s", exc)
        return ""


def _ocr_with_vision(
    content: bytes,
    mime_type: str,
    source_url: str,
    openai_client: Optional[OpenAI],
    gemini_api_key: Optional[str],
    gemini_model: str,
) -> str:
    """
    Envoie un document (image ou PDF scanné) à un LLM vision pour en extraire le texte du menu.
    Essaie Gemini d'abord (supporte PDF natif), puis GPT-4o vision pour les images.
    """
    ocr_prompt = (
        "Ce document est le menu d'un restaurant. "
        "Extrait tout le texte visible: noms des plats, categories, descriptions, prix. "
        "Retourne uniquement le texte brut structure par section."
    )
    b64 = base64.b64encode(content).decode("ascii")

    # Gemini supporte PDF et images nativement
    if gemini_api_key:
        try:
            client = google_genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model=gemini_model,
                contents=[
                    google_genai_types.Part.from_bytes(data=content, mime_type=mime_type),
                    ocr_prompt,
                ],
            )
            text = response.text or ""
            if text.strip():
                logging.info("OCR Gemini reussi pour %s (%s chars)", source_url, len(text))
                return text[:42000]
        except Exception as exc:
            logging.warning("OCR Gemini echec pour %s: %s", source_url, exc)

    # GPT-4o vision pour les images (pas de support PDF natif)
    if openai_client and mime_type.startswith("image/"):
        try:
            response = openai_client.responses.create(
                model="gpt-4o",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": ocr_prompt},
                        {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64}"},
                    ],
                }],
            )
            text = response.output_text or ""
            if text.strip():
                logging.info("OCR GPT-4o reussi pour %s (%s chars)", source_url, len(text))
                return text[:42000]
        except Exception as exc:
            logging.warning("OCR GPT-4o echec pour %s: %s", source_url, exc)

    return ""


def fetch_pdf_or_image_text(
    url: str,
    openai_client: Optional[OpenAI],
    gemini_api_key: Optional[str],
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    timeout: int = 30,
) -> str:
    """
    Télécharge un PDF ou une image et extrait le texte via OCR.
    - PDF avec texte → pdfplumber
    - PDF scanné / image → vision IA (Gemini ou GPT-4o)
    Retourne le texte extrait, ou '' en cas d'échec.
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MenuScraper/1.0)"},
            stream=True,
        )
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get("content-type", "").lower().split(";")[0].strip()
    except Exception as exc:
        logging.warning("Impossible de telecharger %s: %s", url, exc)
        return ""

    ext = Path(urlparse(url).path).suffix.lower()
    is_pdf = ext == ".pdf" or content_type == "application/pdf"
    is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic"} \
        or content_type.startswith("image/")

    if is_pdf:
        text = _extract_pdf_text_pdfplumber(content)
        if text.strip():
            logging.info("PDF texte extrait (pdfplumber) depuis %s (%s chars)", url, len(text))
            return text[:42000]
        logging.info("PDF scanné détecté, bascule sur vision IA: %s", url)
        return _ocr_with_vision(content, "application/pdf", url, openai_client, gemini_api_key, gemini_model)

    if is_image:
        mime = content_type if content_type.startswith("image/") else f"image/{ext.lstrip('.')}"
        return _ocr_with_vision(content, mime, url, openai_client, gemini_api_key, gemini_model)

    return ""


def collect_visible_text_for_restaurant(
    session: requests.Session,
    url: str,
    zyte_api_key: str,
    max_chars: int,
    zyte_timeout: int,
    zyte_retries: int,
    zyte_ip_type: Optional[str],
    zyte_geolocation: Optional[str],
    max_menu_pages: int,
    expand_interactive: bool,
    capture_network: bool,
    openai_client: Optional[OpenAI] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> tuple[str, list[str]]:
    fetched_urls: list[str] = []
    page_texts: list[tuple[str, str]] = []

    first_page = fetch_html_with_zyte(
        session=session,
        url=url,
        zyte_api_key=zyte_api_key,
        timeout=zyte_timeout,
        retries=zyte_retries,
        ip_type=zyte_ip_type,
        geolocation=zyte_geolocation,
        expand_interactive=expand_interactive,
        capture_network=capture_network,
    )

    first_text = clean_html_to_visible_text(first_page.html, max_chars=max_chars)
    if first_page.network_text:
        first_text = f"{first_text}\n\n{first_page.network_text}"

    page_texts.append((first_page.final_url, first_text))
    fetched_urls.append(first_page.final_url)

    # Découverte des liens menu : on inclut plus de candidats pour gérer PDF/images
    candidates = discover_menu_links(
        first_page.html,
        first_page.final_url,
        limit=max(0, max_menu_pages * 2 - 1),  # double pour avoir du choix
    )

    if candidates:
        logging.info(
            "Liens menu detectes pour %s: %s",
            domain_for_log(url),
            ", ".join(
                f"{'[PDF/IMG] ' if c.is_media else ''}{c.text or c.url} (score={c.score})"
                for c in candidates[:5]
            ),
        )

    seen_urls = {first_page.final_url.rstrip("/"), url.rstrip("/")}
    html_pages_added = 0
    media_pages_added = 0

    for candidate in candidates:
        if html_pages_added + media_pages_added >= max_menu_pages - 1:
            break
        if candidate.url.rstrip("/") in seen_urls:
            continue

        seen_urls.add(candidate.url.rstrip("/"))

        if candidate.is_media:
            # PDF ou image : OCR via vision IA
            if not (openai_client or gemini_api_key):
                logging.info("OCR ignoré (aucune clé IA): %s", candidate.url)
                continue
            logging.info("OCR PDF/image en cours: %s", candidate.url)
            ocr_text = fetch_pdf_or_image_text(
                url=candidate.url,
                openai_client=openai_client,
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
            )
            if ocr_text.strip():
                page_texts.append((candidate.url, f"[OCR] {ocr_text}"))
                fetched_urls.append(candidate.url)
                media_pages_added += 1
            else:
                logging.warning("OCR vide pour %s", candidate.url)
            continue

        # Lien HTML classique
        try:
            candidate_page = fetch_html_with_zyte(
                session=session,
                url=candidate.url,
                zyte_api_key=zyte_api_key,
                timeout=zyte_timeout,
                retries=zyte_retries,
                ip_type=zyte_ip_type,
                geolocation=zyte_geolocation,
                expand_interactive=expand_interactive,
                capture_network=capture_network,
            )
        except Exception as exc:
            logging.warning("Lien menu non exploitable %s: %s", candidate.url, exc)
            continue

        candidate_text = clean_html_to_visible_text(candidate_page.html, max_chars=max_chars)
        if candidate_page.network_text:
            candidate_text = f"{candidate_text}\n\n{candidate_page.network_text}"

        density = menu_density_score(candidate_text)
        logging.info(
            "Page menu ajoutee %s (score densité=%s): %s",
            len(page_texts) + 1,
            density,
            candidate.url,
        )

        page_texts.append((candidate_page.final_url, candidate_text))
        fetched_urls.append(candidate_page.final_url)
        seen_urls.add(candidate_page.final_url.rstrip("/"))
        html_pages_added += 1

    return combine_page_texts(page_texts, max_chars=max_chars), fetched_urls


def parse_menu_with_openai(
    client: OpenAI,
    url: str,
    visible_text: str,
    model: str,
) -> AIFlatMenuExtraction:
    system_prompt = (
        "Tu es un expert en extraction structuree de menus de restaurants a partir de texte brut issu de pages web. "
        "Ta priorite absolue est d'extraire TOUS les plats visibles, meme ceux sans prix. "
        "Tu ne rates aucun plat, aucune categorie, aucune section du menu. "
        "Tu n'inventes rien : ni plat, ni prix, ni description qui n'apparait pas dans le texte. "
        "Tu retournes toujours un JSON strictement conforme au schema demande."
    )

    user_prompt = f"""Analyse ce texte extrait d'un site de restaurant et extrait l'integralite du menu.

URL: {url}

TEXTE EXTRAIT:
---
{visible_text}
---

INSTRUCTIONS D'EXTRACTION:

1. menu_disponible = true si au moins un plat ou une carte est identifiable dans le texte.
   menu_disponible = false uniquement si le texte ne contient clairement aucune information de menu.

2. Pour chaque plat trouve, cree un item avec:
   - categorie_plat: la section/categorie du menu (ex: "Entrees", "Pizzas", "Burgers", "Desserts", "Boissons", "Formules").
     Si aucune categorie n'est explicite, utilise "Menu".
   - nom_plat: le nom exact tel qu'il apparait dans le texte. Ne traduis pas, ne modifie pas.
   - description: les ingredients ou la description courte si presente, sinon null.
   - prix: le prix en euros sous forme de nombre decimal (ex: 12.50). null si le prix n'est pas clairement indique.

3. REGLES DE PRIX:
   - Convertis toujours: "12,50€" → 12.50 | "12.5 EUR" → 12.50 | "12€50" → 12.50
   - Si un prix unique s'applique a tous les plats d'une section, mets ce prix sur chaque plat.
   - Si un plat a plusieurs tailles avec des prix differents (ex: petite/grande), cree un item par taille.
   - Ne mets JAMAIS 0. Si le prix est ambigu ou absent, mets null.

4. INCLURE meme si:
   - Le plat n'a pas de prix → met null pour prix
   - La description est absente → met null pour description
   - Le plat est dans une formule → extrait chaque element de formule comme item avec la categorie "Formule"

5. IGNORER:
   - Horaires, adresses, numeros de telephone
   - Avis clients, notes, commentaires
   - Liens de navigation, boutons, menus de site
   - Mentions legales, CGV, politique de confidentialite
   - Publicites et textes promotionnels sans nom de plat

6. Si le texte contient plusieurs sections de menu (ex: plusieurs pages concatenees), extrait tout.

7. Si menu_disponible=false, explique pourquoi dans le champ erreur (ex: "Page institutionnelle sans menu", "Lien mort", "Contenu non lisible").
""".strip()

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=AIFlatMenuExtraction,
        temperature=0,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise OpenAIError("OpenAI n'a pas retourne de sortie structuree parsable")

    return parsed


def _build_ai_prompts(url: str, visible_text: str) -> tuple[str, str]:
    """Retourne (system_prompt, user_prompt) partages entre OpenAI et Gemini."""
    system_prompt = (
        "Tu es un expert en extraction structuree de menus de restaurants a partir de texte brut issu de pages web. "
        "Ta priorite absolue est d'extraire TOUS les plats visibles, meme ceux sans prix. "
        "Tu ne rates aucun plat, aucune categorie, aucune section du menu. "
        "Tu n'inventes rien : ni plat, ni prix, ni description qui n'apparait pas dans le texte. "
        "Tu retournes toujours un JSON strictement conforme au schema demande."
    )
    user_prompt = f"""Analyse ce texte extrait d'un site de restaurant et extrait l'integralite du menu.

URL: {url}

TEXTE EXTRAIT:
---
{visible_text}
---

INSTRUCTIONS D'EXTRACTION:

1. menu_disponible = true si au moins un plat ou une carte est identifiable dans le texte.
   menu_disponible = false uniquement si le texte ne contient clairement aucune information de menu.

2. Pour chaque plat trouve, cree un item avec:
   - categorie_plat: la section/categorie du menu (ex: "Entrees", "Pizzas", "Burgers", "Desserts", "Boissons", "Formules").
     Si aucune categorie n'est explicite, utilise "Menu".
   - nom_plat: le nom exact tel qu'il apparait dans le texte. Ne traduis pas, ne modifie pas.
   - description: les ingredients ou la description courte si presente, sinon null.
   - prix: le prix en euros sous forme de nombre decimal (ex: 12.50). null si le prix n'est pas clairement indique.

3. REGLES DE PRIX:
   - Convertis toujours: "12,50 EUR" -> 12.50 | "12.5 EUR" -> 12.50 | "12€50" -> 12.50
   - Si un prix unique s'applique a tous les plats d'une section, mets ce prix sur chaque plat.
   - Si un plat a plusieurs tailles avec des prix differents, cree un item par taille.
   - Ne mets JAMAIS 0. Si le prix est ambigu ou absent, mets null.

4. INCLURE meme si:
   - Le plat n'a pas de prix -> met null pour prix
   - La description est absente -> met null pour description
   - Le plat est dans une formule -> extrait chaque element avec la categorie "Formule"

5. IGNORER: horaires, adresses, telephone, avis clients, navigation, mentions legales, CGV.

6. Si le texte contient plusieurs sections de menu, extrait tout.

7. Si menu_disponible=false, explique pourquoi dans le champ erreur.
""".strip()
    return system_prompt, user_prompt


def parse_menu_with_gemini(
    api_key: str,
    url: str,
    visible_text: str,
    timeout: int,
    model: str = DEFAULT_GEMINI_MODEL,
) -> AIFlatMenuExtraction:
    system_prompt, user_prompt = _build_ai_prompts(url, visible_text)
    client = google_genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=google_genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=AIFlatMenuExtraction,
            temperature=0,
        ),
    )
    if not response.text:
        raise ValueError("Gemini n'a retourne aucun contenu JSON.")
    try:
        return AIFlatMenuExtraction.model_validate_json(response.text)
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"Gemini: reponse JSON invalide: {exc}") from exc


def _is_openai_quota_or_timeout(exc: Exception) -> bool:
    """Retourne True si l'erreur OpenAI justifie un basculement vers Gemini."""
    if isinstance(exc, (APITimeoutError, RateLimitError)):
        return True
    # APIError générique avec status 429 (insufficient_quota, etc.)
    if isinstance(exc, APIError) and getattr(exc, "status_code", None) == 429:
        return True
    return False


def parse_menu_with_ai(
    openai_client: OpenAI,
    url: str,
    visible_text: str,
    openai_model: str,
    gemini_api_key: Optional[str],
    gemini_model: str,
    gemini_timeout: int,
    ai_provider: str = "auto",
) -> AIFlatMenuExtraction:
    """
    Orchestre les appels IA selon ai_provider :
      - "openai"  : OpenAI uniquement, pas de fallback
      - "gemini"  : Gemini uniquement, pas d'OpenAI
      - "auto"    : OpenAI d'abord, bascule Gemini si quota/timeout, retry OpenAI en dernier recours
    """
    host = domain_for_log(url)

    # --- Mode Gemini uniquement ---
    if ai_provider == "gemini":
        if not gemini_api_key:
            raise ValueError("ai_provider=gemini mais GEMINI_API_KEY non configuree.")
        logging.info("IA: Gemini uniquement pour %s", host)
        return parse_menu_with_gemini(gemini_api_key, url, visible_text, gemini_timeout, gemini_model)

    # --- Mode OpenAI uniquement ---
    if ai_provider == "openai":
        logging.info("IA: OpenAI uniquement pour %s", host)
        return parse_menu_with_openai(openai_client, url, visible_text, openai_model)

    # --- Mode auto (OpenAI → Gemini fallback) ---
    try:
        return parse_menu_with_openai(openai_client, url, visible_text, openai_model)
    except Exception as openai_exc:
        if not _is_openai_quota_or_timeout(openai_exc):
            raise

        if not gemini_api_key:
            logging.error(
                "OpenAI quota depasse pour %s MAIS GEMINI_API_KEY non configuree. "
                "Ajoutez GEMINI_API_KEY ou choisissez ai_provider=gemini.",
                host,
            )
            raise

        logging.warning(
            "OpenAI indisponible pour %s (code=%s) — bascule sur Gemini",
            host,
            getattr(openai_exc, "status_code", "?"),
        )
        try:
            result = parse_menu_with_gemini(gemini_api_key, url, visible_text, gemini_timeout, gemini_model)
            logging.info("Gemini a pris le relais avec succes pour %s", host)
            return result
        except Exception as gemini_exc:
            logging.warning("Gemini a aussi echoue pour %s (%s) — retry OpenAI", host, gemini_exc)
            return parse_menu_with_openai(openai_client, url, visible_text, openai_model)


def parse_price_value(raw_price: str) -> Optional[float]:
    match = PRICE_PATTERN.search(raw_price)
    if not match:
        return None

    value = match.group(1) or match.group(2)
    if not value:
        return None

    try:
        price = float(value.replace(",", "."))
    except ValueError:
        return None

    if price <= 0 or price > 500:
        return None
    return round(price, 2)


def first_price_in_text(text: str) -> Optional[float]:
    return parse_price_value(text)


def find_price_near_name(visible_text: str, name: str, window: int = 5) -> Optional[float]:
    normalized_name = normalize_for_match(name)
    if not normalized_name:
        return None

    name_tokens = [token for token in normalized_name.split() if len(token) > 2]
    if not name_tokens:
        return None

    lines = [line.strip() for line in visible_text.splitlines() if line.strip()]
    normalized_lines = [normalize_for_match(line) for line in lines]

    for index, normalized_line in enumerate(normalized_lines):
        if normalized_name not in normalized_line and not all(token in normalized_line for token in name_tokens[:3]):
            continue

        for neighbor in range(index, min(len(lines), index + window + 1)):
            price = first_price_in_text(lines[neighbor])
            if price is not None:
                return price

        for neighbor in range(max(0, index - 2), index):
            price = first_price_in_text(lines[neighbor])
            if price is not None:
                return price

    return None


def find_shared_category_price(visible_text: str, category_name: str) -> Optional[float]:
    normalized_category = normalize_for_match(category_name)
    if not normalized_category:
        return None

    category_tokens = [token for token in normalized_category.split() if len(token) > 2]
    lines = [line.strip() for line in visible_text.splitlines() if line.strip()]

    for line in lines:
        normalized_line = normalize_for_match(line)
        if normalized_category not in normalized_line and not any(token in normalized_line for token in category_tokens):
            continue
        price = first_price_in_text(line)
        if price is not None:
            return price

    return None


def normalize_result(menu: AIFlatMenuExtraction, url: str, visible_text: str) -> dict:
    if not menu.menu_disponible:
        return error_result(url, menu.erreur or "Aucun menu exploitable detecte par l'IA.")

    categories_by_name: dict[str, list[dict]] = {}
    category_order: list[str] = []

    for item in menu.items:
        category_name = (item.categorie_plat or "Menu").strip() or "Menu"
        dish_name = item.nom_plat.strip()
        if not dish_name:
            continue

        # Tente de trouver un prix fiable, mais garde le plat meme sans prix
        price = item.prix if item.prix and item.prix > 0 else None
        if price is None:
            price = find_price_near_name(visible_text, dish_name)
        if price is None:
            price = find_shared_category_price(visible_text, category_name)
        if price is not None and price <= 0:
            price = None

        if category_name not in categories_by_name:
            categories_by_name[category_name] = []
            category_order.append(category_name)

        categories_by_name[category_name].append(
            {
                "nom_plat": dish_name,
                "description_ingredients": item.description.strip() if item.description else None,
                "prix_euros": round(float(price), 2) if price is not None else None,
            }
        )

    categories = [
        {"nom_categorie": category_name, "plats": categories_by_name[category_name]}
        for category_name in category_order
        if categories_by_name[category_name]
    ]

    if not categories:
        return error_result(
            url,
            menu.erreur or "Menu detecte, mais aucun plat n'a pu etre extrait.",
        )

    return {
        "url_restaurant": url,
        "menu_disponible": True,
        "categories": categories,
    }


def process_url(
    index: int,
    total: int,
    url: str,
    session: requests.Session,
    openai_client: OpenAI,
    zyte_api_key: str,
    model: str,
    max_chars: int,
    zyte_timeout: int,
    zyte_retries: int,
    zyte_ip_type: Optional[str],
    zyte_geolocation: Optional[str],
    max_menu_pages: int,
    expand_interactive: bool,
    capture_network: bool,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_timeout: int = 60,
    ai_provider: str = "auto",
) -> ProcessOutcome:
    host = domain_for_log(url)

    if looks_like_unsupported_resource(url):
        logging.warning("[%s/%s] Ressource PDF/image ignoree: %s", index, total, host)
        return ProcessOutcome(error_result(url, "Ressource PDF/image non supportee."))

    failure_reason = "Erreur inconnue pendant l'extraction."
    try:
        visible_text, fetched_urls = collect_visible_text_for_restaurant(
            session=session,
            url=url,
            zyte_api_key=zyte_api_key,
            max_chars=max_chars,
            zyte_timeout=zyte_timeout,
            zyte_retries=zyte_retries,
            zyte_ip_type=zyte_ip_type,
            zyte_geolocation=zyte_geolocation,
            max_menu_pages=max_menu_pages,
            expand_interactive=expand_interactive,
            capture_network=capture_network,
            openai_client=openai_client,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
        )
        if len(visible_text) < 80:
            logging.warning("[%s/%s] Texte insuffisant pour %s", index, total, host)
            return ProcessOutcome(error_result(url, "Texte insuffisant apres extraction HTML."), fetched_urls)

        logging.info(
            "[%s/%s] HTML extrait pour %s (%s page(s), %s caracteres utiles)",
            index,
            total,
            host,
            len(fetched_urls),
            len(visible_text),
        )

        menu = parse_menu_with_ai(
            openai_client=openai_client,
            url=url,
            visible_text=visible_text,
            openai_model=model,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            gemini_timeout=gemini_timeout,
            ai_provider=ai_provider,
        )
        result = normalize_result(menu, url=url, visible_text=visible_text)
        if not result["menu_disponible"] and result.get("erreur"):
            logging.info("[%s/%s] Erreur IA pour %s: %s", index, total, host, result["erreur"])

        if result["menu_disponible"]:
            dish_count = sum(len(category["plats"]) for category in result["categories"])
            logging.info(
                "[%s/%s] Extraction reussie pour %s: %s categories, %s plats",
                index,
                total,
                host,
                len(result["categories"]),
                dish_count,
            )
        else:
            logging.info("[%s/%s] Aucun menu exploitable pour %s", index, total, host)

        return ProcessOutcome(result, fetched_urls)

    except UnsupportedContentError as exc:
        failure_reason = f"Contenu non supporte: {exc}"
        logging.warning("[%s/%s] Contenu non supporte pour %s: %s", index, total, host, exc)
    except ZyteDownloadError as exc:
        failure_reason = f"Echec Zyte: {exc}"
        level = logging.warning if exc.retryable else logging.error
        level(
            "[%s/%s] Echec Zyte apres retries pour %s: %s",
            index,
            total,
            host,
            exc,
        )
    except DeadPageError as exc:
        failure_reason = f"Page inutilisable: {exc}"
        logging.error("[%s/%s] Page inutilisable pour %s: %s", index, total, host, exc)
    except requests.Timeout:
        failure_reason = "Timeout Zyte."
        logging.error("[%s/%s] Timeout Zyte pour %s", index, total, host)
    except requests.RequestException as exc:
        failure_reason = f"Erreur reseau Zyte: {exc}"
        logging.error("[%s/%s] Erreur reseau Zyte pour %s: %s", index, total, host, exc)
    except (APITimeoutError, RateLimitError, APIError, OpenAIError) as exc:
        failure_reason = f"Erreur OpenAI: {exc}"
        logging.error("[%s/%s] Erreur OpenAI pour %s: %s", index, total, host, exc)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        failure_reason = f"Erreur parsing: {exc}"
        logging.error("[%s/%s] Erreur parsing pour %s: %s", index, total, host, exc)
    except Exception:
        failure_reason = "Erreur inattendue pendant l'extraction."
        logging.exception("[%s/%s] Erreur inattendue pour %s", index, total, host)

    return ProcessOutcome(error_result(url, failure_reason))


def process_url_job(
    index: int,
    total: int,
    url: str,
    args: argparse.Namespace,
    zyte_api_key: str,
    openai_api_key: str,
    gemini_api_key: Optional[str] = None,
) -> ProcessOutcome:
    session = requests.Session()
    openai_client = OpenAI(
        api_key=openai_api_key,
        timeout=args.openai_timeout,
        max_retries=2,
    )

    try:
        return process_url(
            index=index,
            total=total,
            url=url,
            session=session,
            openai_client=openai_client,
            zyte_api_key=zyte_api_key,
            model=args.model,
            max_chars=args.max_chars,
            zyte_timeout=args.zyte_timeout,
            zyte_retries=args.zyte_retries,
            zyte_ip_type=args.zyte_ip_type,
            zyte_geolocation=args.zyte_geolocation,
            max_menu_pages=args.max_menu_pages,
            expand_interactive=not args.no_interactive_expand,
            capture_network=not args.no_network_capture,
            gemini_api_key=gemini_api_key,
            gemini_model=getattr(args, "gemini_model", DEFAULT_GEMINI_MODEL),
            gemini_timeout=getattr(args, "gemini_timeout", 60),
            ai_provider=getattr(args, "ai_provider", "auto"),
        )
    finally:
        session.close()


def persist_progress(
    outcome: ProcessOutcome,
    results: list[dict],
    processed_urls: set[str],
    url: str,
    urls: list[str],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    upsert_result(results, outcome.result)
    sort_results_by_input_order(results, urls)
    processed_urls.add(url)

    save_results_atomic(output_path, results)
    if args.discovered_urls:
        save_discovered_urls_atomic(Path(args.discovered_urls), outcome.discovered_urls)
    if args.csv_output:
        save_flat_csv(Path(args.csv_output), results)
    if args.dashboard:
        save_dashboard(
            Path(args.dashboard),
            results=results,
            urls=urls,
            output_path=output_path,
            log_file=args.log_file,
        )


def result_has_incomplete_prices(result: dict) -> bool:
    if not result.get("menu_disponible"):
        return False

    dish_count = 0
    for category in result.get("categories", []) or []:
        for dish in category.get("plats", []) or []:
            dish_count += 1
            try:
                price = float(dish.get("prix_euros", 0))
            except (TypeError, ValueError):
                return True
            if price <= 0:
                return True

    return dish_count == 0


def count_dishes(result: dict) -> int:
    return sum(
        len(category.get("plats", []) or [])
        for category in result.get("categories", []) or []
        if isinstance(category, dict)
    )


_CSV_FIELDNAMES = [
    "url_restaurant",
    "nom_categorie",
    "nom_plat",
    "description_ingredients",
    "prix_euros",
]


def _write_csv_rows_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def save_flat_csv(path: Path, results: list[dict]) -> None:
    """
    Génère deux CSV automatiquement :
    - <path>               : tous les plats (avec et sans prix)
    - <path stem>_prix_manquants.csv : uniquement les plats sans prix
    """
    if not path:
        return

    all_rows: list[dict] = []
    for result in results:
        if not result.get("menu_disponible"):
            continue
        for category in result.get("categories", []) or []:
            for dish in category.get("plats", []) or []:
                prix = dish.get("prix_euros")
                all_rows.append(
                    {
                        "url_restaurant": result.get("url_restaurant", ""),
                        "nom_categorie": category.get("nom_categorie", ""),
                        "nom_plat": dish.get("nom_plat", ""),
                        "description_ingredients": dish.get("description_ingredients") or "",
                        "prix_euros": prix if prix is not None else "",
                    }
                )

    _write_csv_rows_atomic(path, all_rows)
    logging.info("CSV complet: %s plats -> %s", len(all_rows), path)

    # CSV secondaire : plats sans prix (pour revue manuelle)
    sans_prix = [row for row in all_rows if row["prix_euros"] == ""]
    if sans_prix:
        sans_prix_path = path.with_name(path.stem + "_prix_manquants" + path.suffix)
        _write_csv_rows_atomic(sans_prix_path, sans_prix)
        logging.info(
            "CSV prix manquants: %s plats sans prix -> %s",
            len(sans_prix),
            sans_prix_path,
        )


def read_log_tail(log_file: Optional[str], max_lines: int = 80) -> list[str]:
    if not log_file:
        return []
    path = Path(log_file)
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    except OSError:
        return []


def save_dashboard(
    path: Path,
    results: list[dict],
    urls: list[str],
    output_path: Path,
    log_file: Optional[str],
) -> None:
    if not path:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(urls)
    processed = len({result.get("url_restaurant") for result in results if result.get("url_restaurant")})
    success = sum(1 for result in results if result.get("menu_disponible"))
    failed = sum(1 for result in results if result.get("menu_disponible") is False)
    dish_total = sum(count_dishes(result) for result in results)
    incomplete = sum(1 for result in results if result_has_incomplete_prices(result))
    progress = round((processed / total) * 100, 1) if total else 0

    rows = []
    for result in results[-200:]:
        url = result.get("url_restaurant", "")
        status = "OK" if result.get("menu_disponible") else "A revoir"
        dish_count = count_dishes(result)
        classes = "ok" if result.get("menu_disponible") else "fail"
        if result_has_incomplete_prices(result):
            classes += " warn"
            status = "Prix incomplets"
        rows.append(
            "<tr>"
            f"<td><a href=\"{escape(url)}\" target=\"_blank\">{escape(domain_for_log(url))}</a></td>"
            f"<td><span class=\"pill {classes}\">{escape(status)}</span></td>"
            f"<td>{dish_count}</td>"
            f"<td>{len(result.get('categories', []) or [])}</td>"
            f"<td>{escape(str(result.get('erreur', '') or ''))}</td>"
            "</tr>"
        )

    log_lines = "\n".join(escape(line) for line in read_log_tail(log_file))
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Suivi scraping menus</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, sans-serif; }}
    body {{ margin: 0; background: #f5f7fb; color: #18202f; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 26px; margin: 0 0 6px; }}
    .muted {{ color: #607089; font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 22px 0; }}
    .stat {{ background: white; border: 1px solid #dfe5ef; border-radius: 8px; padding: 16px; }}
    .stat strong {{ display: block; font-size: 26px; margin-top: 6px; }}
    .bar {{ height: 12px; background: #dfe5ef; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; width: {progress}%; background: #1f8a70; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dfe5ef; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid #edf1f7; text-align: left; font-size: 14px; }}
    th {{ background: #edf1f7; color: #354258; }}
    a {{ color: #1463b8; text-decoration: none; }}
    .pill {{ border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 700; }}
    .ok {{ background: #dff7ea; color: #17663b; }}
    .fail {{ background: #ffe6e0; color: #9d2d1d; }}
    .warn {{ background: #fff4cc; color: #7a5200; }}
    pre {{ max-height: 360px; overflow: auto; background: #111827; color: #e5e7eb; padding: 16px; border-radius: 8px; font-size: 12px; }}
    @media (max-width: 900px) {{ .stats {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 18px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Suivi scraping menus</h1>
  <div class="muted">Derniere mise a jour: {generated_at} - JSON: {escape(str(output_path))}</div>
  <section class="stats">
    <div class="stat">Progression<strong>{processed}/{total}</strong></div>
    <div class="stat">Menus trouves<strong>{success}</strong></div>
    <div class="stat">A revoir<strong>{failed}</strong></div>
    <div class="stat">Plats exportes<strong>{dish_total}</strong></div>
    <div class="stat">Prix incomplets<strong>{incomplete}</strong></div>
  </section>
  <div class="bar" aria-label="Progression"><span></span></div>
  <h2>Derniers resultats</h2>
  <table>
    <thead><tr><th>Restaurant</th><th>Statut</th><th>Plats</th><th>Categories</th><th>Erreur</th></tr></thead>
    <tbody>{''.join(rows) or '<tr><td colspan="5">Aucun resultat pour le moment.</td></tr>'}</tbody>
  </table>
  <h2>Logs</h2>
  <pre>{log_lines}</pre>
</main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrait des menus de restaurants via Zyte + OpenAI Structured Outputs."
    )
    parser.add_argument("--urls", default="urls.txt", help="Fichier contenant une URL par ligne.")
    parser.add_argument("--output", default="menus_lyon.json", help="Fichier JSON de sortie.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modele OpenAI a utiliser.")
    parser.add_argument(
        "--ai-provider",
        choices=["auto", "openai", "gemini"],
        default="auto",
        help="IA a utiliser: auto (OpenAI avec fallback Gemini), openai, ou gemini.",
    )
    parser.add_argument("--max-chars", type=int, default=42000, help="Texte max envoye a OpenAI.")
    parser.add_argument("--max-menu-pages", type=int, default=6, help="Pages menu candidates a crawler par URL (inclut PDF/images).")
    parser.add_argument("--zyte-timeout", type=int, default=90, help="Timeout Zyte en secondes.")
    parser.add_argument("--zyte-retries", type=int, default=3, help="Retries Zyte pour bans/rate limits.")
    parser.add_argument(
        "--zyte-ip-type",
        choices=["datacenter", "residential"],
        default=os.getenv("ZYTE_IP_TYPE"),
        help="Optionnel: force ipType Zyte. Exemple: residential.",
    )
    parser.add_argument(
        "--zyte-geolocation",
        default=os.getenv("ZYTE_GEOLOCATION"),
        help="Optionnel: force une geolocalisation Zyte, ex: FR.",
    )
    parser.add_argument("--openai-timeout", type=int, default=90, help="Timeout OpenAI en secondes.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Pause entre deux URLs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Nombre de restaurants traites en meme temps. Recommande: 2 a 4.",
    )
    parser.add_argument("--log-file", default="scrape_menus.log", help="Fichier de logs.")
    parser.add_argument("--dashboard", default="dashboard.html", help="Dashboard HTML local.")
    parser.add_argument("--csv-output", default="menus_lyon.csv", help="Export CSV aplati optionnel.")
    parser.add_argument(
        "--discovered-urls",
        default="nouveauurl.txt",
        help="Fichier TXT des URLs finales/menu detectees pendant le crawl.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Garde les doublons du fichier URLs au lieu de les ignorer.",
    )
    parser.add_argument(
        "--no-interactive-expand",
        action="store_true",
        help="Desactive les actions Zyte qui scrollent/cliquent les onglets de menu.",
    )
    parser.add_argument(
        "--no-network-capture",
        action="store_true",
        help="Desactive la capture des reponses JSON reseau chargees par les widgets menu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess toutes les URLs meme si elles existent deja dans la sortie.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reprocess uniquement les URLs deja sauvegardees avec menu_disponible=false.",
    )
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Reprocess les menus deja sauvegardes qui contiennent des prix a 0 ou aucun plat.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Modele Gemini utilise en fallback si OpenAI timeout.",
    )
    parser.add_argument(
        "--gemini-timeout",
        type=int,
        default=60,
        help="Timeout Gemini en secondes.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(args.log_file)

    zyte_api_key = os.getenv("ZYTE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY") or None

    if not zyte_api_key:
        logging.error("Variable ZYTE_API_KEY manquante dans .env ou l'environnement.")
        return 1
    if not openai_api_key:
        logging.error("Variable OPENAI_API_KEY manquante dans .env ou l'environnement.")
        return 1
    if args.zyte_ip_type and args.zyte_ip_type not in {"datacenter", "residential"}:
        logging.error("ZYTE_IP_TYPE invalide: %s. Valeurs: datacenter ou residential.", args.zyte_ip_type)
        return 1
    if args.workers < 1:
        logging.error("--workers doit etre superieur ou egal a 1.")
        return 1

    urls_path = Path(args.urls)
    output_path = Path(args.output)

    try:
        urls = read_urls(urls_path, keep_duplicates=args.keep_duplicates)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 1

    if not urls:
        logging.error("Aucune URL a traiter dans %s", urls_path)
        return 1

    results = [] if args.overwrite else load_existing_results(output_path)
    processed_urls = set()
    for item in results:
        if not isinstance(item, dict) or not item.get("url_restaurant"):
            continue
        if args.retry_failed and item.get("menu_disponible") is False:
            continue
        if args.retry_incomplete and result_has_incomplete_prices(item):
            continue
        processed_urls.add(item["url_restaurant"])

    total = len(urls)
    jobs: list[tuple[int, str]] = []

    for index, url in enumerate(urls, start=1):
        if not args.overwrite and url in processed_urls:
            logging.info("[%s/%s] Deja traite, ignore: %s", index, total, domain_for_log(url))
            continue
        jobs.append((index, url))

    logging.info(
        "Demarrage: %s URLs totales, %s a traiter, %s worker(s), sortie=%s",
        total,
        len(jobs),
        args.workers,
        output_path,
    )

    if args.workers == 1:
        for job_number, (index, url) in enumerate(jobs, start=1):
            outcome = process_url_job(
                index=index,
                total=total,
                url=url,
                args=args,
                zyte_api_key=zyte_api_key,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
            )

            persist_progress(
                outcome=outcome,
                results=results,
                processed_urls=processed_urls,
                url=url,
                urls=urls,
                output_path=output_path,
                args=args,
            )
            logging.info("[%s/%s] Resultat sauvegarde dans %s", index, total, output_path)

            if args.sleep > 0 and job_number < len(jobs):
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending: dict = {}

            def _drain_completed() -> None:
                done = [f for f in list(pending) if f.done()]
                for f in done:
                    idx, u = pending.pop(f)
                    try:
                        outcome = f.result()
                    except Exception:
                        logging.exception("[%s/%s] Erreur worker pour %s", idx, total, domain_for_log(u))
                        outcome = ProcessOutcome(empty_result(u))
                    persist_progress(
                        outcome=outcome,
                        results=results,
                        processed_urls=processed_urls,
                        url=u,
                        urls=urls,
                        output_path=output_path,
                        args=args,
                    )
                    logging.info("[%s/%s] Resultat sauvegarde dans %s", idx, total, output_path)

            for job_number, (index, url) in enumerate(jobs, start=1):
                future = executor.submit(
                    process_url_job,
                    index,
                    total,
                    url,
                    args,
                    zyte_api_key,
                    openai_api_key,
                    gemini_api_key,
                )
                pending[future] = (index, url)
                if args.sleep > 0 and job_number < len(jobs):
                    time.sleep(args.sleep)
                # Persist any completed results immediately so nothing is lost
                _drain_completed()

            for future in as_completed(pending):
                index, url = pending[future]
                try:
                    outcome = future.result()
                except Exception:
                    logging.exception("[%s/%s] Erreur worker pour %s", index, total, domain_for_log(url))
                    outcome = ProcessOutcome(empty_result(url))
                persist_progress(
                    outcome=outcome,
                    results=results,
                    processed_urls=processed_urls,
                    url=url,
                    urls=urls,
                    output_path=output_path,
                    args=args,
                )
                logging.info("[%s/%s] Resultat sauvegarde dans %s", index, total, output_path)

    logging.info("Termine: %s resultats sauvegardes dans %s", len(results), output_path)
    sort_results_by_input_order(results, urls)
    save_results_atomic(output_path, results)
    if args.discovered_urls:
        save_discovered_urls_atomic(Path(args.discovered_urls), [])
    if args.csv_output:
        save_flat_csv(Path(args.csv_output), results)
    if args.dashboard:
        save_dashboard(
            Path(args.dashboard),
            results=results,
            urls=urls,
            output_path=output_path,
            log_file=args.log_file,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
