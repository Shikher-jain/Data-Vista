import requests
import time
import re
import os
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple

try:
    import contractions
except ModuleNotFoundError:
    contractions = None

from urllib.parse import urljoin, urlparse, urldefrag

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def normalize_cache_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    path = re.sub(r"/+$", "", path)
    if not path:
        path = "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def get_site_name(url: str) -> str:
    netloc = urlparse(url).netloc
    return re.sub(r"[^\w]+", "_", netloc)


def get_qna_jsonl_path(url: str, qna_folder: str = "QnA", leading_dot: bool = True) -> str:
    site_name = get_site_name(url)
    filename = f".{site_name}.jsonl" if leading_dot else f"{site_name}.jsonl"
    return os.path.join(qna_folder, filename)


def get_qna_jsonl_candidates(url: str, qna_folder: str = "QnA") -> List[str]:
    return [
        get_qna_jsonl_path(url, qna_folder=qna_folder, leading_dot=True),
        get_qna_jsonl_path(url, qna_folder=qna_folder, leading_dot=False),
    ]


def _read_jsonl_records(file_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    if not os.path.exists(file_path):
        return records

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
            except json.JSONDecodeError:
                continue
    return records


def load_cached_faqs_for_url(url: str, qna_folder: str = "QnA") -> List[Dict[str, str]]:
    normalized_target = normalize_cache_url(url)

    for file_path in get_qna_jsonl_candidates(url, qna_folder=qna_folder):
        records = _read_jsonl_records(file_path)
        if not records:
            continue

        has_source_urls = any("source_url" in row for row in records)
        if has_source_urls:
            matched = [
                {
                    "question": str(row.get("question", "")).strip(),
                    "answer": str(row.get("answer", "")).strip(),
                }
                for row in records
                if normalize_cache_url(str(row.get("source_url", ""))) == normalized_target
            ]
            matched = [row for row in matched if row["question"] and row["answer"]]
            if matched:
                return deduplicate_faqs(matched)
        else:
            legacy_rows = [
                {
                    "question": str(row.get("question", "")).strip(),
                    "answer": str(row.get("answer", "")).strip(),
                }
                for row in records
            ]
            legacy_rows = [row for row in legacy_rows if row["question"] and row["answer"]]
            if legacy_rows:
                return deduplicate_faqs(legacy_rows)

    return []


def save_faqs_for_url_jsonl(url: str, faqs: List[Dict[str, str]], qna_folder: str = "QnA") -> Tuple[int, str]:
    os.makedirs(qna_folder, exist_ok=True)

    file_path = get_qna_jsonl_path(url, qna_folder=qna_folder, leading_dot=True)
    normalized_target = normalize_cache_url(url)
    clean_faqs = deduplicate_faqs(faqs)

    existing_records = _read_jsonl_records(file_path)
    retained_records: List[Dict[str, str]] = []
    for row in existing_records:
        source_url = str(row.get("source_url", "")).strip()
        if source_url and normalize_cache_url(source_url) == normalized_target:
            continue
        retained_records.append(row)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    new_records = [
        {
            "source_url": normalized_target,
            "question": faq["question"],
            "answer": faq["answer"],
            "extracted_at": timestamp,
        }
        for faq in clean_faqs
    ]

    all_records = retained_records + new_records
    with open(file_path, "w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(new_records), file_path

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

def fetch_url(url: str, timeout: int = 15) -> str:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        res.raise_for_status()
        ctype = res.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype and not url.endswith((".html", ".htm", "/")):
            return ""   
        return res.text
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return ""


def faq_signal_score(html: str) -> int:
    """Estimate how likely a page contains FAQ content."""
    if not html:
        return 0

    lower_html = html.lower()
    score = 0

    for keyword in [
        "faqpage",
        "frequently asked questions",
        "accordion",
        "help center",
        "knowledge base",
        "qna",
        "q&a",
    ]:
        if keyword in lower_html:
            score += 2

    score += lower_html.count("?") // 3

    try:
        soup = BeautifulSoup(html, "html.parser")
        score += len(soup.find_all("details"))
        score += len(soup.find_all("summary"))
        score += len(soup.find_all("dt"))
        score += len(soup.find_all("dd"))

        for script in soup.find_all("script", type=lambda t: t and "ld+json" in t):
            if script.string and "FAQPage" in script.string:
                score += 5
    except Exception:
        pass

    return score


def fetch_url_playwright(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Render JS-heavy pages via Playwright when available."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return "", f"Playwright not available: {exc}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0")
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            # Expand common FAQ containers.
            try:
                page.evaluate("""
                    () => {
                        document.querySelectorAll('details').forEach((d) => { d.open = true; });
                        document.querySelectorAll('[aria-expanded="false"]').forEach((el) => {
                            try { el.click(); } catch (_) {}
                        });
                    }
                """)
            except Exception:
                pass

            html = page.content()
            browser.close()
            return html, ""
    except Exception as exc:
        return "", f"Playwright render failed: {exc}"


def fetch_url_selenium(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Render JS-heavy pages via Selenium when available."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except Exception as exc:
        return "", f"Selenium not available: {exc}"

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("user-agent=Mozilla/5.0")

        # Requires chromedriver available in PATH.
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        try:
            driver.execute_script(
                """
                document.querySelectorAll('details').forEach((d) => { d.open = true; });
                document.querySelectorAll('[aria-expanded="false"]').forEach((el) => {
                    try { el.click(); } catch (_) {}
                });
                """
            )
        except Exception:
            pass

        return driver.page_source, ""
    except Exception as exc:
        return "", f"Selenium render failed: {exc}"
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


def fetch_url_with_fallback(url: str, timeout: int = 20, allow_dynamic: bool = True) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
    """Fetch HTML using static requests first, then dynamic renderers if needed."""
    report: Dict[str, List[Dict[str, str]]] = {"attempts": []}

    best_html = fetch_url(url, timeout=timeout)
    best_method = "requests"
    best_score = faq_signal_score(best_html)
    report["attempts"].append(
        {
            "method": "requests",
            "success": "true" if bool(best_html) else "false",
            "score": str(best_score),
            "error": "",
        }
    )

    # If static fetch already looks promising, avoid heavy browser startup.
    if not allow_dynamic or best_score >= 5:
        report["selected"] = [{"method": best_method, "score": str(best_score)}]
        return best_html, report

    for method_name, method_fn in [
        ("playwright", fetch_url_playwright),
        ("selenium", fetch_url_selenium),
    ]:
        dynamic_html, dynamic_error = method_fn(url, timeout=timeout)
        dynamic_score = faq_signal_score(dynamic_html)
        report["attempts"].append(
            {
                "method": method_name,
                "success": "true" if bool(dynamic_html) else "false",
                "score": str(dynamic_score),
                "error": dynamic_error,
            }
        )

        if dynamic_html and (dynamic_score > best_score or (dynamic_score == best_score and len(dynamic_html) > len(best_html))):
            best_html = dynamic_html
            best_score = dynamic_score
            best_method = method_name

    report["selected"] = [{"method": best_method, "score": str(best_score)}]
    return best_html, report


def extract_faqs_from_url(
    url: str,
    max_follow_links: int = 6,
    timeout: int = 20,
    allow_dynamic: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    """Extract FAQs from a URL and a handful of likely FAQ/support child pages."""
    summary: Dict[str, List[Dict[str, str]]] = {
        "attempts": [],
        "pages": [],
        "warnings": [],
    }

    root_html, root_report = fetch_url_with_fallback(url, timeout=timeout, allow_dynamic=allow_dynamic)
    summary["attempts"].extend(root_report.get("attempts", []))
    summary["pages"].append({"url": url, "faqs": str(0)})

    if not root_html:
        summary["warnings"].append({"message": "Could not fetch the target page HTML."})
        return [], summary

    all_faqs = extract_faqs_from_html(root_html)
    summary["pages"][0]["faqs"] = str(len(all_faqs))

    # Follow likely FAQ links when direct extraction is weak.
    candidate_links = extract_links(root_html, url)
    if candidate_links and max_follow_links > 0:
        for idx, link in enumerate(candidate_links[:max_follow_links]):
            if link == url:
                continue

            allow_dynamic_for_link = allow_dynamic and len(all_faqs) < 5 and idx < 3
            child_html, child_report = fetch_url_with_fallback(link, timeout=timeout, allow_dynamic=allow_dynamic_for_link)
            summary["attempts"].extend(child_report.get("attempts", []))

            if not child_html:
                summary["warnings"].append({"message": f"Skipping unreadable linked page: {link}"})
                continue

            child_faqs = extract_faqs_from_html(child_html)
            summary["pages"].append({"url": link, "faqs": str(len(child_faqs))})
            all_faqs.extend(child_faqs)

    deduped = deduplicate_faqs(all_faqs)
    return deduped, summary


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

def process_page(url: str, base_url: str, timeout: int = 20, allow_dynamic: bool = False):
    if allow_dynamic:
        html, _ = fetch_url_with_fallback(url, timeout=timeout, allow_dynamic=True)
    else:
        html = fetch_url(url, timeout=timeout)

    if not html:
        return [], []

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

    return links, faqs


def crawl_site(
    root_url: str,
    max_depth: int,
    max_workers: int,
    timeout: int = 20,
    allow_dynamic: bool = False,
    max_pages: int = 80,
):
    base = root_url
    seen = set()
    seen_lock = Lock()
    all_urls = []
    all_faqs = []

    frontier = [root_url]

    for depth in range(max_depth + 1):
        if not frontier:
            break

        if len(all_urls) >= max_pages:
            break

        this_batch = []
        with seen_lock:
            for u in frontier:
                if u not in seen:
                    seen.add(u)
                    this_batch.append(u)

        remaining_budget = max_pages - len(all_urls)
        if remaining_budget <= 0:
            break
        this_batch = this_batch[:remaining_budget]

        if not this_batch:
            break

        next_frontier = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(process_page, u, base, timeout, allow_dynamic): u
                for u in this_batch
            }
            for fut in as_completed(future_map):
                u = future_map[fut]
                try:
                    links, faqs = fut.result()
                    all_urls.append(u)
                    all_faqs.extend(faqs)
                    next_frontier.extend(links)
                except Exception as e:
                    print(f"[WARN] Failed processing {u}: {e}")
                    pass

        # Deduplicate next frontier
        frontier = list(dict.fromkeys([x for x in next_frontier if x not in seen and same_domain(x, base)]))

    # Deduplicate final FAQs by question + answer
    all_faqs = deduplicate_faqs(all_faqs)
    return list(dict.fromkeys(all_urls)), all_faqs

def main():
    start_url = input("Enter URL: ")

    qna_folder = "QnA"
    os.makedirs(qna_folder, exist_ok=True)

    qna_file = get_qna_jsonl_path(start_url, qna_folder=qna_folder, leading_dot=True)

    cached_rows = load_cached_faqs_for_url(start_url, qna_folder=qna_folder)
    if cached_rows:
        print(f"FAQs already extracted for {start_url}.")
        print(f"Use existing file: {qna_file}")
        return

    try:
        max_depth = int(input("Enter crawl depth (default 3): ") or 3)
    except ValueError:
        max_depth = 3
    try:
        max_workers = int(input("Enter max workers (default 12): ") or 12)
    except ValueError:
        max_workers = 12

    print("Running crawler...")
    t0 = time.time()
    urls, faqs = crawl_site(start_url, max_depth, max_workers)
    dt = round(time.time() - t0, 2)

    print(f"\nDone in {dt} seconds")
    print(f"Crawled {len(urls)} pages")
    print(f"Extracted {len(faqs)} FAQs\n")

    try:
        min_len = int(input("Min answer length (0-2000, default 20): ") or 20)
    except ValueError:
        min_len = 20

    filtered = [f for f in faqs if len(f["answer"]) >= min_len]

    saved_count, saved_path = save_faqs_for_url_jsonl(start_url, filtered, qna_folder=qna_folder)
    print(f"Saved {saved_count} QnA FAQs : {saved_path}")

if __name__ == "__main__":
    main()
