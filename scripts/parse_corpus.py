#!/usr/bin/env python3
"""
Parse and scrape the complete Aquinas corpus from HTML sources.
Outputs a unified chunks.jsonl file for Qdrant indexing.
"""

import json
import os
import re
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

# Base paths
WRITINGS_DIR = Path(__file__).parent.parent / "writings"
SCRAPED_DIR = WRITINGS_DIR / "scraped"
OUTPUT_DIR = Path(__file__).parent.parent / "out"

# Rate limiting for scraping
REQUEST_DELAY = 0.5  # seconds between requests

# Work metadata
WORKS = {
    "summa_theologica": {
        "name": "Summa Theologica",
        "abbrev": "ST",
        "source_file": "Summa Theologica.html",
    },
    "contra_gentiles": {
        "name": "Summa Contra Gentiles",
        "abbrev": "SCG",
        "source_file": "Thomas Aquinas_ Contra Gentiles_ English.html",
    },
    "de_ente": {
        "name": "On Being and Essence",
        "abbrev": "DBE",
        "source_file": "Thomas Aquinas_ De ente et essentia_ English.html",
    },
    "de_anima": {
        "name": "Quaestiones de Anima",
        "abbrev": "QDA",
        "source_file": "the_soul.html",
    },
    "de_potentia": {
        "name": "De Potentia Dei",
        "abbrev": "DPD",
        "source_file": "de_potentia_dei.html",
    },
    "de_veritate": {
        "name": "De Veritate",
        "abbrev": "DV",
        "source_file": "Veritate.html",
    },
    "metaphysics": {
        "name": "Commentary on Metaphysics",
        "abbrev": "CM",
        "source_file": "Thomas Aquinas_ Commentary on Aristotle's Metaphysics_ English.html",
    },
}


def get_cache_path(url: str) -> Path:
    """Get cache file path for a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    filename = urlparse(url).path.split("/")[-1] or "index.html"
    return SCRAPED_DIR / f"{url_hash}_{filename}"


def fetch_url(url: str, use_cache: bool = True) -> str:
    """Fetch URL content with caching using curl."""
    cache_path = get_cache_path(url)
    
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    
    print(f"  Fetching: {url}")
    time.sleep(REQUEST_DELAY)
    
    # Use curl with -k to skip SSL verification (old server)
    result = subprocess.run(
        ["curl", "-k", "-s", "-L", "--max-time", "30", url],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"curl failed: {result.stderr}")
    
    content = result.stdout
    
    # Cache the response
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content, encoding="utf-8")
    
    return content


def extract_links(html: str, base_url: str = "") -> list[str]:
    """Extract all http/https links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            links.append(href)
        elif href.startswith("/") and base_url:
            links.append(urljoin(base_url, href))
        elif not href.startswith("#") and base_url:
            links.append(urljoin(base_url, href))
    return list(set(links))


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def extract_english_from_table(table) -> str:
    """Extract English text from Latin/English parallel table."""
    texts = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            # English is typically in the second cell
            english_cell = cells[1]
            text = clean_text(english_cell.get_text())
            if text:
                texts.append(text)
        elif len(cells) == 1:
            # Single cell - might be a header or English-only
            text = clean_text(cells[0].get_text())
            if text:
                texts.append(text)
    return "\n\n".join(texts)


# =============================================================================
# PARSERS FOR EACH WORK
# =============================================================================

def parse_summa_theologica(html: str, source_url: str = "") -> Generator[dict, None, None]:
    """Parse Summa Theologica from HTML."""
    soup = BeautifulSoup(html, "lxml")
    work_info = WORKS["summa_theologica"]
    
    # Find all article anchors (format: FPQ1A1THEP1, FPQ1A2THEP1, etc.)
    anchors = soup.find_all("a", {"name": re.compile(r"^[A-Z]+Q\d+A\d+")})
    
    if not anchors:
        # Try to find content directly from tables
        tables = soup.find_all("table")
        for i, table in enumerate(tables):
            text = extract_english_from_table(table)
            if len(text) > 100:
                yield {
                    "id": f"ST_section_{i}",
                    "work": work_info["name"],
                    "work_abbrev": work_info["abbrev"],
                    "location": f"Section {i}",
                    "section_type": "content",
                    "title": "",
                    "text": text,
                    "source_url": source_url,
                    "language": "en",
                }
        return
    
    for anchor in anchors:
        anchor_name = anchor.get("name", "")
        
        # Parse anchor name (e.g., FPQ1A1THEP1)
        match = re.match(r"([A-Z]+)Q(\d+)A(\d+)", anchor_name)
        if not match:
            continue
            
        part_code, question, article = match.groups()
        
        # Find the next table after this anchor
        next_table = anchor.find_next("table")
        if not next_table:
            continue
            
        # Extract article title (usually in h3)
        title_elem = anchor.find_next("h3")
        title = clean_text(title_elem.get_text()) if title_elem else ""
        
        # Extract text from table
        text = extract_english_from_table(next_table)
        
        if len(text) > 50:
            # Determine part name
            part_names = {
                "FP": "First Part",
                "FS": "First Part of Second Part", 
                "SS": "Second Part of Second Part",
                "TP": "Third Part",
                "XP": "Supplement",
            }
            part_name = part_names.get(part_code, part_code)
            
            yield {
                "id": f"ST_{part_code}_Q{question}_A{article}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"{part_name}, Question {question}, Article {article}",
                "section_type": "article",
                "title": title,
                "text": text,
                "source_url": source_url,
                "language": "en",
            }


def parse_contra_gentiles(html: str, source_url: str = "") -> Generator[dict, None, None]:
    """Parse Summa Contra Gentiles from HTML."""
    soup = BeautifulSoup(html, "lxml")
    work_info = WORKS["contra_gentiles"]
    
    # Find chapter anchors (format: <a name="1">, <a name="2">, etc.)
    anchors = soup.find_all("a", {"name": re.compile(r"^\d+$")})
    
    # Determine which book this is from the title/header
    book_match = re.search(r"BOOK\s+(ONE|TWO|THREE|FOUR|I+V?)", html, re.IGNORECASE)
    book_num = "1"
    if book_match:
        book_map = {"ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", 
                    "I": "1", "II": "2", "III": "3", "IV": "4"}
        book_num = book_map.get(book_match.group(1).upper(), "1")
    
    for anchor in anchors:
        chapter = anchor.get("name", "")
        
        # Find the next table after this anchor
        next_table = anchor.find_next("table")
        if not next_table:
            continue
        
        # Get chapter title from the table
        title_row = next_table.find("tr")
        title = ""
        if title_row:
            title_cells = title_row.find_all("td")
            if len(title_cells) >= 2:
                title = clean_text(title_cells[1].get_text())
        
        # Extract text from table
        text = extract_english_from_table(next_table)
        
        if len(text) > 50:
            yield {
                "id": f"SCG_B{book_num}_C{chapter}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Book {book_num}, Chapter {chapter}",
                "section_type": "chapter",
                "title": title,
                "text": text,
                "source_url": source_url,
                "language": "en",
            }


def parse_de_ente(html: str, source_url: str = "") -> Generator[dict, None, None]:
    """Parse De Ente et Essentia (On Being and Essence) from HTML."""
    soup = BeautifulSoup(html, "lxml")
    work_info = WORKS["de_ente"]
    
    # Find all paragraphs - they're numbered 1, 2, 3...
    paragraphs = soup.find_all("p")
    
    current_chunk = []
    chunk_start = 1
    
    for p in paragraphs:
        text = clean_text(p.get_text())
        
        # Skip empty or very short paragraphs
        if len(text) < 20:
            continue
            
        # Check if starts with a number (paragraph marker)
        num_match = re.match(r"^(\d+)\.", text)
        
        current_chunk.append(text)
        
        # Create chunk every 5 paragraphs or ~2000 chars
        total_len = sum(len(t) for t in current_chunk)
        if len(current_chunk) >= 5 or total_len > 2000:
            chunk_end = chunk_start + len(current_chunk) - 1
            yield {
                "id": f"DBE_para_{chunk_start}_{chunk_end}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Paragraphs {chunk_start}-{chunk_end}",
                "section_type": "paragraphs",
                "title": "",
                "text": "\n\n".join(current_chunk),
                "source_url": source_url,
                "language": "en",
            }
            chunk_start = chunk_end + 1
            current_chunk = []
    
    # Emit remaining
    if current_chunk:
        chunk_end = chunk_start + len(current_chunk) - 1
        yield {
            "id": f"DBE_para_{chunk_start}_{chunk_end}",
            "work": work_info["name"],
            "work_abbrev": work_info["abbrev"],
            "location": f"Paragraphs {chunk_start}-{chunk_end}",
            "section_type": "paragraphs",
            "title": "",
            "text": "\n\n".join(current_chunk),
            "source_url": source_url,
            "language": "en",
        }


def parse_de_anima(html: str, source_url: str = "") -> Generator[dict, None, None]:
    """Parse Quaestiones de Anima (The Soul) from HTML."""
    soup = BeautifulSoup(html, "lxml")
    work_info = WORKS["de_anima"]
    
    # Find article anchors (format: <a name="1">, <a name="2">, etc.)
    anchors = soup.find_all("a", {"name": re.compile(r"^\d+$")})
    
    for anchor in anchors:
        article_num = anchor.get("name", "")
        
        # Find article title
        title_elem = anchor.find_next("p", style=re.compile(r"font-weight.*bold"))
        title = ""
        if title_elem:
            title = clean_text(title_elem.get_text())
        
        # Collect all paragraphs until next anchor
        next_anchor = anchor.find_next("a", {"name": re.compile(r"^\d+$")})
        
        paragraphs = []
        current = anchor.find_next("p")
        while current and (not next_anchor or current.sourceline < getattr(next_anchor, 'sourceline', float('inf'))):
            text = clean_text(current.get_text())
            if len(text) > 20:
                paragraphs.append(text)
            current = current.find_next_sibling("p")
            if not current:
                current = current if current else anchor.parent.find_next("p") if anchor.parent else None
                break
        
        if paragraphs:
            yield {
                "id": f"QDA_A{article_num}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Article {article_num}",
                "section_type": "article",
                "title": title,
                "text": "\n\n".join(paragraphs),
                "source_url": source_url,
                "language": "en",
            }


def parse_disputata_question(html: str, work_key: str, question_num: str, source_url: str = "") -> Generator[dict, None, None]:
    """Parse a Quaestiones Disputatae question (De Potentia, De Veritate)."""
    soup = BeautifulSoup(html, "lxml")
    work_info = WORKS[work_key]
    
    # Find all article sections - anchors like "1:1", "1:2", "2:1" or just "1", "2"
    articles = soup.find_all("a", {"name": re.compile(r"^\d+:\d+$|^\d+$")})
    
    if not articles:
        # Try to get all content as one chunk from tables
        tables = soup.find_all("table")
        all_text = []
        for table in tables:
            text = extract_english_from_table(table)
            if len(text) > 50:
                all_text.append(text)
        
        if all_text:
            yield {
                "id": f"{work_info['abbrev']}_Q{question_num}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Question {question_num}",
                "section_type": "question",
                "title": "",
                "text": "\n\n".join(all_text)[:8000],
                "source_url": source_url,
                "language": "en",
            }
        return
    
    # Get list of all anchor names for boundary detection
    anchor_names = [a.get("name", "") for a in articles]
    
    for i, anchor in enumerate(articles):
        article_id = anchor.get("name", "")
        
        # Find article title (usually in bold or header nearby)
        title_elem = anchor.find_next(["b", "strong", "h3", "h4"])
        title = ""
        if title_elem:
            title = clean_text(title_elem.get_text())
        
        # Find the next anchor to determine boundary
        next_anchor = articles[i + 1] if i + 1 < len(articles) else None
        
        # Collect text from tables between this anchor and the next
        texts = []
        current_table = anchor.find_next("table")
        
        while current_table:
            # Check if we've passed the next anchor
            if next_anchor:
                # Simple check: if current table is after next anchor in DOM order
                try:
                    next_anchor_pos = str(soup).find(str(next_anchor))
                    current_table_pos = str(soup).find(str(current_table))
                    if current_table_pos > next_anchor_pos:
                        break
                except:
                    pass
            
            text = extract_english_from_table(current_table)
            if len(text) > 20:
                texts.append(text)
            
            current_table = current_table.find_next_sibling("table")
            if not current_table:
                current_table = current_table.find_next("table") if current_table else None
            
            if len(texts) > 20:  # Safety limit
                break
        
        if texts:
            yield {
                "id": f"{work_info['abbrev']}_Q{question_num}_A{article_id.replace(':', '_')}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Question {question_num}, Article {article_id}",
                "section_type": "article",
                "title": title,
                "text": "\n\n".join(texts)[:8000],
                "source_url": source_url,
                "language": "en",
            }


def parse_metaphysics_book(html: str, book_num: str, source_url: str = "") -> Generator[dict, None, None]:
    """Parse a book from Commentary on Metaphysics."""
    soup = BeautifulSoup(html, "lxml")
    work_info = WORKS["metaphysics"]
    
    # Find lessons (usually marked with headers or specific patterns)
    # Look for "Lesson" headers
    lessons = soup.find_all(string=re.compile(r"Lesson\s+\d+", re.IGNORECASE))
    
    if not lessons:
        # Try to chunk by paragraphs
        paragraphs = soup.find_all("p")
        chunk = []
        chunk_num = 1
        
        for p in paragraphs:
            text = clean_text(p.get_text())
            if len(text) > 20:
                chunk.append(text)
            
            if len(chunk) >= 10 or sum(len(t) for t in chunk) > 3000:
                yield {
                    "id": f"CM_B{book_num}_chunk{chunk_num}",
                    "work": work_info["name"],
                    "work_abbrev": work_info["abbrev"],
                    "location": f"Book {book_num}, Section {chunk_num}",
                    "section_type": "section",
                    "title": "",
                    "text": "\n\n".join(chunk),
                    "source_url": source_url,
                    "language": "en",
                }
                chunk = []
                chunk_num += 1
        
        if chunk:
            yield {
                "id": f"CM_B{book_num}_chunk{chunk_num}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Book {book_num}, Section {chunk_num}",
                "section_type": "section",
                "title": "",
                "text": "\n\n".join(chunk),
                "source_url": source_url,
                "language": "en",
            }
        return
    
    # Process by lessons
    for i, lesson_match in enumerate(lessons):
        lesson_elem = lesson_match.parent if lesson_match.parent else lesson_match
        lesson_num = re.search(r"Lesson\s+(\d+)", str(lesson_match), re.IGNORECASE)
        lesson_num = lesson_num.group(1) if lesson_num else str(i + 1)
        
        # Get text after lesson header
        texts = []
        current = lesson_elem
        next_lesson = lessons[i + 1].parent if i + 1 < len(lessons) else None
        
        while current:
            current = current.find_next(["p", "td"])
            if not current:
                break
            if next_lesson and current == next_lesson:
                break
            text = clean_text(current.get_text())
            if len(text) > 20 and "Lesson" not in text[:20]:
                texts.append(text)
        
        if texts:
            yield {
                "id": f"CM_B{book_num}_L{lesson_num}",
                "work": work_info["name"],
                "work_abbrev": work_info["abbrev"],
                "location": f"Book {book_num}, Lesson {lesson_num}",
                "section_type": "lesson",
                "title": "",
                "text": "\n\n".join(texts)[:8000],
                "source_url": source_url,
                "language": "en",
            }


# =============================================================================
# MAIN SCRAPING/PARSING LOGIC
# =============================================================================

def scrape_and_parse_work(work_key: str) -> Generator[dict, None, None]:
    """Scrape and parse a single work."""
    work_info = WORKS[work_key]
    source_file = WRITINGS_DIR / work_info["source_file"]
    
    if not source_file.exists():
        print(f"  Warning: Source file not found: {source_file}")
        return
    
    html = source_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    
    # Extract base URL for resolving relative links
    base_url_match = re.search(r'saved from url=\([\d]+\)([^">\)]+)', html)
    base_url = base_url_match.group(1) if base_url_match else ""
    
    # Get all external links
    all_links = extract_links(html, base_url)
    isidore_links = [l for l in all_links if "isidore.co/aquinas" in l]
    
    print(f"  Found {len(isidore_links)} external links to scrape")
    
    # If no external links, parse the file directly
    if not isidore_links:
        print(f"  Parsing self-contained HTML...")
        if work_key == "summa_theologica":
            yield from parse_summa_theologica(html, str(source_file))
        elif work_key == "contra_gentiles":
            yield from parse_contra_gentiles(html, str(source_file))
        elif work_key == "de_ente":
            yield from parse_de_ente(html, str(source_file))
        elif work_key == "de_anima":
            yield from parse_de_anima(html, str(source_file))
        return
    
    # Scrape and parse external links
    for link in tqdm(isidore_links, desc=f"  Scraping {work_info['abbrev']}"):
        try:
            link_html = fetch_url(link)
            
            # Determine which parser to use based on link pattern
            if work_key == "de_potentia":
                q_match = re.search(r"QDdePotentia(\d+)", link)
                if q_match:
                    yield from parse_disputata_question(link_html, work_key, q_match.group(1), link)
            
            elif work_key == "de_veritate":
                q_match = re.search(r"QDdeVer(\d+)", link)
                if q_match:
                    yield from parse_disputata_question(link_html, work_key, q_match.group(1), link)
            
            elif work_key == "metaphysics":
                b_match = re.search(r"Metaphysics(\d+)", link)
                if b_match:
                    yield from parse_metaphysics_book(link_html, b_match.group(1), link)
            
            elif work_key == "summa_theologica":
                yield from parse_summa_theologica(link_html, link)
            
            elif work_key == "contra_gentiles":
                yield from parse_contra_gentiles(link_html, link)
            
            else:
                # Generic parsing
                body = BeautifulSoup(link_html, "lxml").find("body")
                if body:
                    text = clean_text(body.get_text())
                    if len(text) > 100:
                        link_id = urlparse(link).path.split("/")[-1].replace(".htm", "").replace(".html", "")
                        yield {
                            "id": f"{work_info['abbrev']}_{link_id}",
                            "work": work_info["name"],
                            "work_abbrev": work_info["abbrev"],
                            "location": link_id,
                            "section_type": "page",
                            "title": "",
                            "text": text[:8000],
                            "source_url": link,
                            "language": "en",
                        }
        
        except Exception as e:
            print(f"  Error processing {link}: {e}")
            continue


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Aquinas corpus")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR / "chunks.jsonl"),
                        help="Output JSONL file path")
    parser.add_argument("--works", "-w", nargs="+", choices=list(WORKS.keys()) + ["all"],
                        default=["all"], help="Which works to parse")
    parser.add_argument("--no-cache", action="store_true", 
                        help="Don't use cached scraped pages")
    args = parser.parse_args()
    
    # Determine works to process
    works_to_process = list(WORKS.keys()) if "all" in args.works else args.works
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each work
    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for work_key in works_to_process:
            print(f"\nProcessing: {WORKS[work_key]['name']}")
            work_chunks = 0
            
            for chunk in scrape_and_parse_work(work_key):
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                work_chunks += 1
            
            print(f"  Generated {work_chunks} chunks")
            total_chunks += work_chunks
    
    print(f"\n{'='*50}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
