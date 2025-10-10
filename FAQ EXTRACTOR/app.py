import requests
import time
import re
import os
import json
from bs4 import BeautifulSoup

import contractions

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

def save_faqs_jsonl(faqs, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for faq in faqs:
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()
            if q and a:
            
                record = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]
                }    
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
def same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc

def normalize_url(href: str, base: str) -> str:
    absolute = urljoin(base, href)
    absolute, _ = urldefrag(absolute)  # remove (#fragment)
    return absolute

def filter_links_by_keywords(links):
    return [u for u in links if any(k in u.lower() for k in FAQ_KEYWORDS_IN_URL)]
def remove_abb(text):
    return contractions.fix(text)

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

def process_page(url: str, base_url: str):
    html = fetch_url(url)
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
def crawl_site(root_url: str, max_depth: int, max_workers: int):
    base = root_url
    seen = set()
    seen_lock = Lock()
    all_urls = []
    all_faqs = []

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
            future_map = {ex.submit(process_page, u, base): u for u in this_batch}
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

    def get_site_name(url):
        netloc = urlparse(url).netloc
        return re.sub(r"[^\w]+", "_", netloc)

    site_name = get_site_name(start_url)

    qna_folder = "QnA"
    ft_folder = "FineTuning"
    os.makedirs(qna_folder, exist_ok=True)
    os.makedirs(ft_folder, exist_ok=True)

    qna_file = os.path.join(qna_folder, f"{site_name}.jsonl")
    ft_file = os.path.join(ft_folder, f"{site_name}.jsonl")

    if os.path.exists(qna_file) and os.path.exists(ft_file):
        print(f"FAQs already extracted for {start_url}.")
        print(f"Use existing files: {qna_file} and {ft_file}")
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

    with open(qna_file, "w", encoding="utf-8") as f:
        for faq in filtered:
            f.write(json.dumps(faq, ensure_ascii=False) + "\n")

    save_faqs_jsonl(filtered, ft_file)

    print(f"Saved QnA FAQs : {qna_file}")
    print(f"Saved Fine-tuning FAQs : {ft_file}")

if __name__ == "__main__":
    main()
