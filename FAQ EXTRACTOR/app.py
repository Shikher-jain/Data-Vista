import requests
import time
import re
import os
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from bs4 import BeautifulSoup

try:
    import contractions
except ModuleNotFoundError:
    contractions = None

from urllib.parse import urljoin, urlparse, urldefrag

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, abort, render_template, request, send_file

try:
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    sync_playwright = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
except ModuleNotFoundError:
    webdriver = None
    ChromeOptions = None

FAQ_KEYWORDS_IN_URL = [
    "faq", "faqs", "help", "support", "customer-service",
    "knowledge", "knowledgebase", "kb", "guide",
    "how-to", "howto", "troubleshoot", "troubleshooting",
    "qna", "q&a", "questions", "getting-started", "contact-us"
]

FAQ_TEXT_HINTS = [
    "faq", "frequently asked questions", "how do i", "how to",
    "troubleshoot", "troubleshooting", "common questions",
    "help center", "support", "customer service"
]

PLACEHOLDER_KEYWORDS = ["click here", "learn more", "more info", "link", "reference"]

QUESTION_PATTERN = re.compile(r"\b(what|how|when|where|why|which|who|do|does|did|can|should|is|are|will|there|any)\b.*\?", re.I)

BASE_DIR = Path(__file__).resolve().parent
QNA_FOLDER = BASE_DIR / "QnA"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))


def get_site_name(url):
    netloc = urlparse(url).netloc
    return re.sub(r"[^\w]+", "_", netloc)


def load_jsonl_faqs(path):
    faqs = []
    if not Path(path).exists():
        return faqs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                faqs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return faqs


def save_jsonl_faqs(path, faqs):
    with open(path, "w", encoding="utf-8") as f:
        for faq in faqs:
            f.write(json.dumps(faq, ensure_ascii=False) + "\n")


def _build_fetch_attempt(method: str, success: bool, score: int = 0, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "method": method,
        "success": success,
        "score": int(score),
        "error": error or "",
    }


def _signal_score(html_text: str) -> int:
    if not html_text:
        return 0

    score = 0
    lower = html_text.lower()

    if len(html_text) > 2000:
        score += 10
    if len(html_text) > 10000:
        score += 10
    if "application/ld+json" in lower:
        score += 15
    if "faq" in lower:
        score += 15
    if any(hint in lower for hint in FAQ_TEXT_HINTS):
        score += 20
    if "accordion" in lower or "<details" in lower:
        score += 15
    if "question" in lower and "answer" in lower:
        score += 10

    return min(score, 100)


def fetch_url_with_fallback(url: str, timeout: int = 15, allow_dynamic: bool = False) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    attempts: List[Dict[str, Any]] = []
    best_html = ""
    best_score = -1
    last_error: Optional[str] = None

    def consider(method: str, html_text: str, error: Optional[str] = None) -> None:
        nonlocal best_html, best_score
        score = _signal_score(html_text)
        attempts.append(_build_fetch_attempt(method, bool(html_text), score, error))
        if html_text and score >= best_score:
            best_html = html_text
            best_score = score

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        html_text = response.text if ("text/html" in content_type or url.endswith((".html", ".htm", "/"))) else ""
        consider("requests", html_text, None if html_text else f"Unsupported content type: {content_type or 'unknown'}")
    except Exception as exc:
        last_error = str(exc)
        consider("requests", "", last_error)

    should_try_dynamic = allow_dynamic and (not best_html or best_score < 25)
    if should_try_dynamic:
        if sync_playwright is not None:
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page(viewport={"width": 1440, "height": 1800})
                    page.set_default_timeout(max(timeout, 1) * 1000)
                    page.goto(url, wait_until="networkidle")
                    html_text = page.content()
                    browser.close()
                consider("playwright", html_text, None if html_text else "Playwright returned empty HTML")
            except Exception as exc:
                consider("playwright", "", str(exc))

        if not best_html and webdriver is not None and ChromeOptions is not None:
            try:
                options = ChromeOptions()
                options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                driver = webdriver.Chrome(options=options)
                try:
                    driver.set_page_load_timeout(max(timeout, 1))
                    driver.get(url)
                    html_text = driver.page_source or ""
                finally:
                    driver.quit()
                consider("selenium", html_text, None if html_text else "Selenium returned empty HTML")
            except Exception as exc:
                consider("selenium", "", str(exc))

    if best_html:
        return best_html, attempts, None

    if last_error:
        return "", attempts, last_error

    return "", attempts, "No HTML content could be fetched."


def run_extraction(
    start_url: str,
    max_depth: int,
    max_workers: int,
    timeout: int = 15,
    min_answer_len: int = 20,
    reuse_cache: bool = True,
    max_pages: int | None = None,
    allow_dynamic: bool = True,
):
    site_name = get_site_name(start_url)
    QNA_FOLDER.mkdir(parents=True, exist_ok=True)

    qna_file = QNA_FOLDER / f"{site_name}.jsonl"

    if reuse_cache and qna_file.exists():
        cached_faqs = [faq for faq in load_jsonl_faqs(qna_file) if len(str(faq.get("answer", "")).strip()) >= min_answer_len]
        return {
            "site_name": site_name,
            "urls": [],
            "faqs": cached_faqs,
            "page_count": 0,
            "faq_count": len(cached_faqs),
            "filtered_count": len(cached_faqs),
            "raw_count": len(cached_faqs),
            "duration": 0.0,
            "qna_file": qna_file,
            "cached": True,
            "fetch_attempts": [],
            "warnings": [],
        }

    t0 = time.time()
    urls, faqs, fetch_attempts = crawl_site(
        start_url,
        max_depth,
        max_workers,
        timeout=timeout,
        allow_dynamic=allow_dynamic,
        max_pages=max_pages,
    )
    duration = round(time.time() - t0, 2)

    filtered = [f for f in faqs if len(f.get("answer", "")) >= min_answer_len]
    save_jsonl_faqs(qna_file, filtered)

    warnings = [
        item.get("error", "").strip()
        for item in fetch_attempts
        if str(item.get("error", "")).strip()
    ]
    if not filtered and not warnings:
        warnings = ["Fetched pages did not expose supported FAQ structures."]

    return {
        "site_name": site_name,
        "urls": urls,
        "faqs": filtered,
        "page_count": len(urls),
        "faq_count": len(filtered),
        "filtered_count": len(filtered),
        "raw_count": len(faqs),
        "duration": duration,
        "qna_file": qna_file,
        "cached": False,
        "fetch_attempts": fetch_attempts,
        "warnings": warnings,
    }


def clean_answer(text, all_questions):
    if not text:
        return ""
    
    sentences = re.split(r'(?<=[.?!])\s+', text)  # split by sentence
    final = []
    for s in sentences:
        # Stop if this sentence looks like a new question
        if QUESTION_PATTERN.search(s):
            break
        # Stop if sentence contains any known question explicitly
        if any(q.lower() in s.lower() for q in all_questions):
            break
        final.append(s)
    return " ".join(final).strip()

def looks_like_question(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    # Accept if it ends with a question mark
    if text.endswith("?"):
        return True

    # Accept if it starts with common question words or Q-number style
    return bool(
        re.match(
            r"^(q[\.\-\:\)]*\s*\d*\s*|what|how|why|when|where|which|who|can|could|should|may|"
            r"is|are|will|there|any|do|does|did|have|has|had|i\s+am|am\s+i)\b",
            text,
            re.I,
        )
    )

def fetch_url(url: str, timeout: int = 15, allow_dynamic: bool = False) -> str:
    html, _, _ = fetch_url_with_fallback(url, timeout=timeout, allow_dynamic=allow_dynamic)
    return html
def same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc

def normalize_url(href: str, base: str) -> str:
    absolute = urljoin(base, href)
    absolute, _ = urldefrag(absolute)  # remove (#fragment)
    return absolute

def filter_links_by_keywords(links):
    return [u for u in links if any(k in u.lower() for k in FAQ_KEYWORDS_IN_URL)]
def remove_abb(text):
    if contractions is None:
        return text
    try:
        return contractions.fix(text)
    except Exception:
        return text

def clean_text(text: str) -> str:
    text = re.sub(r"¶", "", text)
    text = re.sub(r"Â", "", text)
    text = re.sub(r"<\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\xa0", " ", text)   
    text = re.sub(r'^(q[\s\.:;\-\)]*\d*\.?\s*)', '', text, flags=re.I)
    text = re.sub(r'^\d+\.\s*', '', text)  # removes leading numbering like "1. "
    text = re.sub(r'\s+\d+\.\s*$', '', text)  # removes trailing numbering like " 1."
    text = re.sub(r"^\s+|\s+$", "", text)  # removes leading and trailing whitespace
    text = re.sub(r"\[.*?\]", " ", text)  # removes [edit] etc.
    text = remove_abb(text)

    # Remove duplicate sentences
    sentences = re.split(r"(?<=[.?!])\s+", text)
    seen, unique = set(), []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    # Rejoin sentences into cleaned text
    return " ".join(unique).strip()

def extract_links(html: str, base_url: str):
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        u = normalize_url(a["href"], base_url)

        # Skip URLs pointing to other domains
        if not same_domain(u, base_url):
            continue

        # Skip static assets or tracking URLs
        if re.search(r"(\.pdf|\.jpg|\.png|\.gif|\.zip|\.mp4|\.css|\.js|\?.*utm|tracking)", u):
            continue

        # Skip URLs that repeat the domain in the path (trap URLs)
        if urlparse(base_url).netloc in u[len(base_url):]:
            continue

        # Keep only URLs likely related to FAQ/help
        if not any(k in u.lower() for k in FAQ_KEYWORDS_IN_URL):
            continue

        links.append(u)

    return list(dict.fromkeys(links))
    # -----------------------------------------
    # if not html:
    #     return []
    # soup = BeautifulSoup(html, "html.parser")
    # links = []
    # for a in soup.find_all("a", href=True):
    #     u = normalize_url(a["href"], base_url)
    #     if same_domain(u, base_url):
    #         links.append(u)
    # links = [u for u in links if not re.search(r"(\.pdf|\.jpg|\.png|\.gif|\.zip|\.mp4|\.css|\.js|\?.*utm|tracking)", u)]
    # return list(dict.fromkeys(links))

def deduplicate_faqs(faqs):
    final_faqs = []
    seen = set()
    for f in faqs:
        q = clean_text(f.get("question", ""))
        a = clean_text(f.get("answer", ""))
        if len(q) < 5 or len(a) < 5:
            continue

        # key = f"{q}|||{a}"
        key = q.lower()  # deduplicate by question only

        if key not in seen:
            seen.add(key)
            final_faqs.append({"question": q, "answer": clean_answer(a, [q])})
    return final_faqs

def extract_faqs_from_html(html: str):
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script","style","noscript","iframe","template","nav","footer","form","button","input","svg"]):
        tag.decompose()
    faqs = []

    # 1) JSON-LD
    for script in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(script.string or "")
            candidates = data if isinstance(data, list) else [data]

            for d in candidates:
                if not isinstance(d, dict):
                    continue

                # Get @type (can be string or list)
                t = d.get("@type") or d.get("type")
                if t == "FAQPage" or (isinstance(t, list) and "FAQPage" in t):
                    main = d.get("mainEntity") or []
                    for item in main:
                        if not isinstance(item, dict):
                            continue

                        q = item.get("name") or item.get("headline")
                        acc = item.get("acceptedAnswer")
                        ans = acc.get("text") if isinstance(acc, dict) else None

                        if q and ans:
                            faqs.append({
                                "question": clean_text(q),
                                "answer": clean_text(ans)
                            })
        except Exception:
            pass

    # 2) <details>/<summary>
    for det in soup.find_all("details"):
        summary = det.find("summary")
        if summary:
            q = summary.get_text(" ", strip=True)
            content = det.find_all(["p","div","section","article","ul","ol"])
            a = " ".join([clean_text(c.get_text(" ", strip=True)) for c in content]) if content else ""
            if q and (a or q.endswith("?")):
                faqs.append({"question": q, "answer": a})

    # 3) <dl>/<dt>/<dd>
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                q = dt.get_text(" ", strip=True)
                a = dd.get_text(" ", strip=True)
                if q and a:
                    faqs.append({"question": q, "answer": a})

    # 4) Headings
    for h in soup.find_all(["h2","h3","h4","button"]):
        qtxt = h.get_text(" ", strip=True)
        if not qtxt:
            continue

        if looks_like_question(qtxt):
            nxt = h.find_next_sibling(lambda tag: tag.name in ["p","div","section","article","ul","ol"])
            a = clean_text(nxt.get_text(" ", strip=True)) if nxt else ""
            if qtxt and a:
                faqs.append({"question": qtxt, "answer": a})

    # 4) Accordion items (common in many sites)
    for block in soup.find_all("div", class_=lambda x: x and "accordion-item" in x):
        q_tag = block.find(["h2","h3","h4","button"])
        a_tag = block.find(["p","div","section","article","ul","ol"])

        if q_tag and a_tag:
            q = clean_text(q_tag.get_text(" ", strip=True))
            a = clean_text(a_tag.get_text(" ", strip=True))

            if q and a:
                faqs.append({
                    "question": q,
                    "answer": a
                })

    # 5) Accordion style (common in AWS/Flipkart)
    for block in soup.find_all("div", class_=lambda x: x and "accordion" in x.lower()):
        q_tag = block.find(["h2","h3","h4","button"])
        a_tag = block.find(["p","div","section","article","ul","ol"])
        if q_tag and a_tag:
            q = clean_text(q_tag.get_text(" ", strip=True))
            a = clean_text(a_tag.get_text(" ", strip=True))
            if q and a:
                faqs.append({"question": q, "answer": a})

    # 6) Tables (Q in first <td>, A in second <td>)
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) == 2:
            q = clean_text(tds[0].get_text(" ", strip=True))
            a = clean_text(tds[1].get_text(" ", strip=True))
            if q.endswith("?") and a:
                faqs.append({"question": q, "answer": a})

    # 7)  Q:/A:
    blocks = [clean_text(t.get_text(" ", strip=True)) for t in soup.find_all(["p","div","li","span","strong","b"])]
    q, a = None, []

    for t in blocks:
        if not t:
            continue
        if t.endswith("?") or re.match(r"^\s*q[:\-]\s*", t, re.I):
            if q and a:
                faqs.append({"question": q, "answer": " ".join(a)})
            q, a = re.sub(r"^\s*q[:\-]\s*", "", t, flags=re.I), []
        else:
            if q is not None:
                if re.match(r"^\s*a[:\-]\s*", t, re.I):
                    t = re.sub(r"^\s*a[:\-]\s*", "", t, flags=re.I)
                a.append(t)
    
    # 8) FAQ sections (h2/h3 as section, h4/p/li as Q/A)
    for h in soup.select("h3"):
        q = clean_text(h.get_text(" ", strip=True))
        if not q.endswith("?"):
            continue
        # Answer is next sibling or next block until something that looks like next question
        nxt = h.find_next_sibling()
        answer_text = ""
        # Collect paragraphs / lists until next h3
        while nxt and nxt.name not in ["h3"]:
            if nxt.name in ["p","div","ul","ol"]:
                answer_text += " " + nxt.get_text(" ", strip=True)
            nxt = nxt.find_next_sibling()
        a = clean_text(answer_text)
        if q and a:
            faqs.append({"question": q, "answer": a})

    # 9) Flexible accordion handling (AWS/Flipkart/others)
    for block in soup.find_all("div", class_=lambda x: x and "accordion" in x.lower()):
        q_tag = block.find(["button","h2","h3","h4"], class_=lambda x: not x or "trigger" in (x or "").lower())
        a_tag = block.find(["div","section","p","ul","ol","article"], class_=lambda x: True)
        if q_tag and a_tag:
            q = clean_text(q_tag.get_text(" ", strip=True))
            a = clean_text(a_tag.get_text(" ", strip=True))
            if q and a:
                faqs.append({"question": q, "answer": a})

    # 10) Each topic is a section, e.g., <div id="topic-1"> ... </div>
    topics = soup.select("div.lb-grid")  
    for topic in topics:  
        questions = topic.find_all(["h3","strong"])  
        for q_tag in questions:  
            question = clean_text(q_tag.get_text(" ", strip=True))  
            answer_parts = []  
            for sib in q_tag.find_all_next():  
                if sib.name in ["h3", "strong"]:  
                    break  
                if sib.name in ["p","div","ul","ol","li"]:  
                    answer_parts.append(clean_text(sib.get_text(" ", strip=True)))  
            answer = " ".join(answer_parts).strip()  
            if question and answer:  
                faqs.append({"question": clean_text(question), "answer": clean_text(answer)})  

    # 11) AWS Expandable Section FAQs
    for block in soup.find_all("div", class_=lambda x: x and "itemExpander_module_expandableSection" in x):
        q_tag = block.find(["h2", "h3", "button"])
        a_tag = block.find("div", class_=lambda x: x and "itemExpander_module_expandableSectionContent" in x)
        
        if q_tag and a_tag:
            q = clean_text(q_tag.get_text(" ", strip=True))
            a = clean_text(a_tag.get_text(" ", strip=True))
            if q and a:
                faqs.append({"question": q, "answer": a})

    # if q and a:
    #     faqs.append({"question": clean_text(q), "answer": " ".join(clean_text(a))})

    return deduplicate_faqs(faqs)

def process_page(url: str, base_url: str, timeout: int = 15, allow_dynamic: bool = True):
    html, fetch_attempts, _ = fetch_url_with_fallback(url, timeout=timeout, allow_dynamic=allow_dynamic)
    if not html:
        return [], [], fetch_attempts

    # Extract FAQs from the page
    faqs = extract_faqs_from_html(html)

    # Extract links from the page
    links = extract_links(html, base_url)
    links = [u for u in links if same_domain(u, base_url)]
    links = filter_links_by_keywords(links)

    # Follow placeholder links inside answers
    soup = BeautifulSoup(html, "html.parser")
    for f in faqs:
        soup_ans = BeautifulSoup(f["answer"], "html.parser")
        for a_tag in soup_ans.find_all("a", href=True):
            u = normalize_url(a_tag["href"], base_url)
            if same_domain(u, base_url) and u not in links:
                links.append(u)

    # Deduplicate links
    links = list(dict.fromkeys(links))

    return links, faqs, fetch_attempts
def crawl_site(
    root_url: str,
    max_depth: int,
    max_workers: int,
    timeout: int = 15,
    allow_dynamic: bool = True,
    max_pages: int | None = None,
):
    base = root_url
    seen = set()
    seen_lock = Lock()
    all_urls = []
    all_faqs = []
    all_fetch_attempts = []

    frontier = [root_url]

    for depth in range(max_depth + 1):
        if not frontier:
            break

        this_batch = []
        with seen_lock:
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    this_batch.append(u)

        if not this_batch:
            break

        next_frontier = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(process_page, u, base, timeout, allow_dynamic): u for u in this_batch}
            for fut in as_completed(future_map):
                u = future_map[fut]
                try:
                    links, faqs, fetch_attempts = fut.result()
                    all_urls.append(u)
                    all_faqs.extend(faqs)
                    all_fetch_attempts.extend(fetch_attempts)
                    next_frontier.extend(links)
                except Exception as e:
                    print(f"[WARN] Failed processing {u}: {e}")
                    pass

        if max_pages is not None and len(all_urls) >= max_pages:
            break

        # Deduplicate next frontier
        frontier = list(dict.fromkeys([x for x in next_frontier if x not in seen and same_domain(x, base)]))

    # Deduplicate final FAQs by question + answer
    all_faqs = deduplicate_faqs(all_faqs)
    return list(dict.fromkeys(all_urls)), all_faqs, all_fetch_attempts

def is_valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def run_cli_extraction(start_url: str, max_depth: int, max_workers: int, min_len: int):
    result = run_extraction(start_url, max_depth, max_workers, timeout=15, min_answer_len=min_len)

    if result["cached"]:
        print(f"FAQs already extracted for {start_url}.")
        print(f"Use existing file: {result['qna_file']}")
        return result

    print(f"\nDone in {result['duration']} seconds")
    print(f"Crawled {len(result['urls'])} pages")
    print(f"Extracted {result['raw_count']} FAQs\n")
    for warning in result.get("warnings", []):
        print(f"[WARN] {warning}")
    print(f"Saved QnA FAQs : {result['qna_file']}")
    return result


def main():
    start_url = input("Enter URL: ")

    try:
        max_depth = int(input("Enter crawl depth (default 3): ") or 3)
    except ValueError:
        max_depth = 3

    try:
        max_workers = int(input("Enter max workers (default 12): ") or 12)
    except ValueError:
        max_workers = 12

    try:
        min_len = int(input("Min answer length (0-2000, default 20): ") or 20)
    except ValueError:
        min_len = 20

    run_cli_extraction(start_url, max_depth, max_workers, min_len)


@app.route("/", methods=["GET", "POST"])
def web_ui():
    defaults = {
        "url_input": "",
        "crawl_depth": 3,
        "max_workers": 12,
        "timeout": 15,
        "min_answer_len": 20,
        "message": None,
        "error": None,
        "cached": False,
        "site_name": None,
        "qna_file": None,
        "pages": [],
        "faqs": [],
        "page_count": 0,
        "faq_count": 0,
        "duration": None,
        "fetch_attempts": [],
        "warnings": [],
    }

    if request.method == "GET":
        return render_template("faq_extractor.html", **defaults)

    url_input = (request.form.get("url") or "").strip()
    crawl_depth_raw = (request.form.get("crawl_depth") or "3").strip()
    max_workers_raw = (request.form.get("max_workers") or "12").strip()
    timeout_raw = (request.form.get("timeout") or "15").strip()
    min_answer_len_raw = (request.form.get("min_answer_len") or "20").strip()

    try:
        crawl_depth = max(0, min(5, int(crawl_depth_raw)))
        max_workers = max(1, min(20, int(max_workers_raw)))
        timeout = max(5, min(90, int(timeout_raw)))
        min_answer_len = max(0, min(3000, int(min_answer_len_raw)))
    except ValueError:
        return render_template(
            "faq_extractor.html",
            **defaults,
            url_input=url_input,
            error="Crawler depth, max workers, timeout, and minimum answer length must be whole numbers.",
        )

    if not url_input:
        return render_template("faq_extractor.html", **defaults, url_input=url_input, error="Please provide a URL.")
    if not is_valid_http_url(url_input):
        return render_template(
            "faq_extractor.html",
            **defaults,
            url_input=url_input,
            error="Please provide a valid URL starting with http:// or https://.",
        )

    try:
        result = run_extraction(url_input, crawl_depth, max_workers, timeout=timeout, min_answer_len=min_answer_len)
    except Exception as exc:
        return render_template(
            "faq_extractor.html",
            **defaults,
            url_input=url_input,
            crawl_depth=crawl_depth,
            max_workers=max_workers,
            timeout=timeout,
            min_answer_len=min_answer_len,
            error=f"Extraction failed: {exc}",
        )

    pages = [{"url": url} for url in result["urls"]]
    fetch_attempts = result.get("fetch_attempts", [])
    warnings = [warning for warning in result.get("warnings", []) if warning]
    error_text = None
    if not result["faqs"]:
        error_text = "No FAQ pairs were detected for the provided URL."
        if warnings:
            error_text = f"{error_text} Details: {' | '.join(warnings[:2])}"
    message = (
        f"Loaded {result['faq_count']} FAQs from cache for {result['site_name']}."
        if result["cached"]
        else f"Done in {result['duration']} seconds. Crawled {result['page_count']} pages and extracted {result['faq_count']} FAQs."
    )

    return render_template(
        "faq_extractor.html",
        url_input=url_input,
        crawl_depth=crawl_depth,
        max_workers=max_workers,
        timeout=timeout,
        min_answer_len=min_answer_len,
        message=message,
        error=error_text,
        cached=result["cached"],
        site_name=result["site_name"],
        qna_file=str(result["qna_file"]),
        pages=pages,
        faqs=result["faqs"][:30],
        page_count=result["page_count"],
        faq_count=result["faq_count"],
        duration=result["duration"],
        fetch_attempts=fetch_attempts,
        warnings=warnings,
    )


@app.route("/download/<path:filename>")
def download_qna(filename):
    file_path = QNA_FOLDER / filename
    if not file_path.exists() or file_path.is_dir():
        abort(404)
    return send_file(file_path, as_attachment=True, download_name=file_path.name)


def run_web_app():
    app.run(host="127.0.0.1", port=5055, debug=False, use_reloader=False)


if __name__ == "__main__":
    if "--cli" in sys.argv:
        main()
    else:
        run_web_app()
