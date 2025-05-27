import requests, json, re
from bs4 import BeautifulSoup
from collections import deque
from datetime import datetime
import pandas as pd

BASE_URL = "https://www.thairath.co.th"
NEWS_URL = f"{BASE_URL}/news/"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
DATE_RE  = re.compile(r"\d{1,2}\s[ก-๙]+\.\s\d{4}\s\d{1,2}:\d{2}")

def yield_weekly_links(news_tag: str, filter: str, seen: set[str]):
    page = 1
    while True:
        url = f"{BASE_URL}/news/{news_tag}/all-latest?filter={filter}&page={page}"
        print(f"🔄 Fetching {url} …")

        html   = requests.get(url, headers=HEADERS, timeout=15).text
        soup   = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if script is None:                          # page structure changed?
            print("❌  __NEXT_DATA__ not found")
            break

        data = json.loads(script.string)

        # the list of stories lives in two possible branches;
        # 'news' is present on section pages, 'common' on some others
        try:
            items = data["props"]["initialState"]["news"]["data"]["items"]
        except KeyError:
            items = data["props"]["initialState"]["common"]["data"]["items"]

        if not items:                               # reached the end
            break

        fresh = 0
        for it in items:
            # prefer canonical; fall back to fullPath and make it absolute
            print(f"  • {it.get('title', '').strip()} → {it.get('canonical', '') or it.get('fullPath', '')}")
            link  = it.get("fullPath", "")
            title = it.get("title", "").strip()

            if not link or link in seen:
                continue
            seen.add(link)
            fresh += 1
            yield title, link

        # stop early if this page gave nothing new
        if fresh == 0:
            break

        # or stop when we hit the last page, which you can read from the blob too
        max_page = data["props"]["initialState"]["common"]["data"]["page"]["max"]
        if page >= max_page:
            break
        page += 1


def bfs_find(obj, want):
    q = deque([obj])
    while q:
        node = q.popleft()
        hit  = want(node)
        if hit:
            return hit
        if isinstance(node, dict):
            q.extend(node.values())
        elif isinstance(node, list):
            q.extend(node)
    return None


def find_article_blob(obj):
    return bfs_find(obj,
        lambda n: isinstance(n, dict)
        and "content" in n and isinstance(n["content"], list)
    )


def find_tags_list(obj):
    return bfs_find(obj,
        lambda n: isinstance(n, dict)
        and isinstance(n.get("tags"), list)
        and all(isinstance(x, str) for x in n["tags"])
        and n["tags"]
    ) or []


def clean_date(txt):
    if not txt: return "N/A"
    txt = txt.replace("\xa0", " ").strip()
    return txt[:-2].strip() if txt.endswith("น.") else txt


def normalise_tags(raw):
    if not raw: return "N/A"
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            raw = [t.get("title", "") for t in raw]
        return ", ".join(t.strip() for t in raw if t.strip()) or "N/A"
    if isinstance(raw, str):
        return raw.strip() or "N/A"
    return "N/A"

def scrape_article_details(url: str):
    try:
        html  = requests.get(url, headers=HEADERS, timeout=15).text
        soup  = BeautifulSoup(html, "html.parser")

        script = soup.find("script", id="__NEXT_DATA__")
        json_tags, published, content = [], "N/A", "N/A"

        if script:
            data = json.loads(script.string)
            json_tags = find_tags_list(data)

            article_blob = find_article_blob(data)
            if article_blob:
                published = article_blob.get("publishLabelThai", "N/A")
                paragraphs = [
                    b.get("data", {}).get("text", "")
                    for b in article_blob["content"]
                    if b.get("type") == "paragraph"
                ]
                body = "\n\n".join(p.strip() for p in paragraphs if p.strip())
                content = body or content

        if published == "N/A":
            date_div = soup.select_one('div[class*="item_article-date"]')
            if date_div:
                published = clean_date(date_div.get_text(" ", strip=True))
            elif soup.find(text=DATE_RE):
                published = clean_date(soup.find(text=DATE_RE))

        if content == "N/A":
            body_p = (soup.select("div[itemprop='articleBody'] p")
                      or soup.select("div[class*='evs3ejl'] p"))
            content_html = "\n\n".join(p.get_text(strip=True) for p in body_p)
            if content_html.strip():
                content = content_html

        if not json_tags:
            chips = soup.select_one('div[class*="item_article-tags"]')
            if chips:
                json_tags = [a.get_text(strip=True) for a in chips.find_all("a")]

        tags = normalise_tags(json_tags)
        return published, content, tags

    except Exception as e:
        print(f"❌ Failed {url}\n   {e}")
        return "N/A", "N/A", "N/A"

def scrape_thairath(outpath: str | None = None,
                    tags: list[str] = None):
    if tags is None:
        # pick the sections you care about – add/remove at will
        tags = ["local", "society", "politic"]

    print("🔄 Fetching ThaiRath weekly archive …")
    articles, seen = [], set()

    for tag in tags:
        print(f"  • {tag}")
        for title, link in yield_weekly_links(tag, "7", seen):
            full_url = BASE_URL + '/' + link
            pub, body, tag_list = scrape_article_details(full_url)
            articles.append({
                "title":      title,
                "url":        full_url,
                "scraped_at": datetime.now().isoformat(),
                "published":  pub,
                "content":    body,
                "tags":       tag_list,
            })

    df = pd.DataFrame(articles)
    if outpath:
        df.to_csv(outpath, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(df)} rows → {outpath}")
    else:
        fname = f"thairath_{datetime.now():%Y%m%dT%H%M}.csv"
        df.to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(df)} rows → {fname}")