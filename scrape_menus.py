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
from openai import APIError, APITimeoutError, OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ZYTE_ENDPOINT = "https://api.zyte.com/v1/extract"
DEFAULT_MODEL = "gpt-4o-mini"
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
            "header",
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
    if looks_like_unsupported_resource(href):
        score -= 5
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
        if score < 4:
            continue

        existing = candidates_by_url.get(absolute_url)
        if existing is None or score > existing.score:
            candidates_by_url[absolute_url] = MenuLinkCandidate(
                url=absolute_url,
                text=text[:120],
                score=score,
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

    candidates = discover_menu_links(
        first_page.html,
        first_page.final_url,
        limit=max(0, max_menu_pages - 1),
    )

    if candidates:
        logging.info(
            "Liens menu detectes pour %s: %s",
            domain_for_log(url),
            ", ".join(f"{candidate.text or candidate.url} ({candidate.score})" for candidate in candidates[:3]),
        )

    seen_urls = {first_page.final_url.rstrip("/"), url.rstrip("/")}
    for candidate in candidates:
        if len(page_texts) >= max_menu_pages:
            break
        if candidate.url.rstrip("/") in seen_urls:
            continue
        if looks_like_unsupported_resource(candidate.url):
            logging.info("Lien menu ignore car PDF/image: %s", candidate.url)
            continue

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
        if density < 5:
            logging.info(
                "Lien menu garde malgre score faible (%s): %s",
                density,
                candidate.url,
            )

        page_texts.append((candidate_page.final_url, candidate_text))
        fetched_urls.append(candidate_page.final_url)
        seen_urls.add(candidate_page.final_url.rstrip("/"))

    return combine_page_texts(page_texts, max_chars=max_chars), fetched_urls


def parse_menu_with_openai(
    client: OpenAI,
    url: str,
    visible_text: str,
    model: str,
) -> AIFlatMenuExtraction:
    system_prompt = (
        "Tu es un expert en extraction de menus de restaurants depuis du texte HTML nettoye. "
        "Tu retournes uniquement des donnees conformes au schema structure. "
        "Tu n'inventes jamais un plat, un prix, une categorie ou des ingredients."
    )

    user_prompt = f"""
Analyse la page de ce restaurant et extrait le menu sous forme de JSON structure.

URL source: {url}

Texte visible extrait du site:
---
{visible_text}
---

Format attendu:
- menu_disponible=true si un menu exploitable est present.
- items doit etre un tableau d'objets avec exactement:
  categorie_plat: categorie du plat, ex: Entrees, Burgers, Desserts.
  nom_plat: nom exact du plat.
  description: ingredients ou description, null si absent.
  prix: uniquement le nombre du prix en euros, ex: 12.50. Null si absent ou ambigu.
- Si la page ne contient pas de menu, si le lien est mort, si c'est un PDF/image,
  ou si le menu n'est pas exploitable: menu_disponible=false, items=[], erreur explicite.

Regles strictes:
- Ignore horaires, adresses, telephone, avis clients, reseaux sociaux, navigation,
  mentions legales et textes marketing.
- Conserve la langue originale des categories et plats.
- Convertis "12,50 EUR" ou "12,50€" en 12.50.
- Ne mets jamais 0 ou 0.0 pour un prix manquant. Utilise null.
- Si un prix commun s'applique a toute une categorie ou un groupe visible de plats,
  applique ce prix a chaque plat du groupe.
- Si un plat a plusieurs tailles/formules avec prix distincts, cree un item par variante
  et indique la variante dans nom_plat.
- N'inclus pas les supplements seuls comme plats principaux, sauf s'ils sont clairement
  vendus comme articles de menu.
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

        price = item.prix
        if price is None or price <= 0:
            price = find_price_near_name(visible_text, dish_name)
        if price is None or price <= 0:
            shared_category_price = find_shared_category_price(visible_text, category_name)
            if shared_category_price is not None:
                price = shared_category_price
        if price is None or price <= 0:
            logging.debug("Plat ignore sans prix fiable: %s (%s)", dish_name, url)
            continue

        if category_name not in categories_by_name:
            categories_by_name[category_name] = []
            category_order.append(category_name)

        categories_by_name[category_name].append(
            {
                "nom_plat": dish_name,
                "description_ingredients": item.description.strip() if item.description else None,
                "prix_euros": round(float(price), 2),
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
            menu.erreur or "Menu detecte, mais aucun plat avec prix fiable n'a pu etre extrait.",
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

        menu = parse_menu_with_openai(
            client=openai_client,
            url=url,
            visible_text=visible_text,
            model=model,
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


def save_flat_csv(path: Path, results: list[dict]) -> None:
    if not path:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "url_restaurant",
                "nom_categorie",
                "nom_plat",
                "description_ingredients",
                "prix_euros",
            ],
        )
        writer.writeheader()
        for result in results:
            if not result.get("menu_disponible"):
                continue
            for category in result.get("categories", []) or []:
                for dish in category.get("plats", []) or []:
                    writer.writerow(
                        {
                            "url_restaurant": result.get("url_restaurant", ""),
                            "nom_categorie": category.get("nom_categorie", ""),
                            "nom_plat": dish.get("nom_plat", ""),
                            "description_ingredients": dish.get("description_ingredients") or "",
                            "prix_euros": dish.get("prix_euros", ""),
                        }
                    )
    tmp_path.replace(path)


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
    parser.add_argument("--max-chars", type=int, default=30000, help="Texte max envoye a OpenAI.")
    parser.add_argument("--max-menu-pages", type=int, default=3, help="Pages menu candidates a crawler par URL.")
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
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(args.log_file)

    zyte_api_key = os.getenv("ZYTE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

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
            future_to_job = {}
            for job_number, (index, url) in enumerate(jobs, start=1):
                future = executor.submit(
                    process_url_job,
                    index,
                    total,
                    url,
                    args,
                    zyte_api_key,
                    openai_api_key,
                )
                future_to_job[future] = (index, url)
                if args.sleep > 0 and job_number < len(jobs):
                    time.sleep(args.sleep)

            for future in as_completed(future_to_job):
                index, url = future_to_job[future]
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
