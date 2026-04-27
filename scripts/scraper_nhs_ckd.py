"""
NHS CKD Content Scraper
-----------------------
Scrapes all CKD-related pages from the NHS website,
splits them into section-based chunks, and saves as JSON.

Run: pip install requests beautifulsoup4
Then: python scrape_nhs_ckd.py
"""

import requests
import json
import re
import os
from typing import Optional, List
from bs4 import BeautifulSoup
from datetime import datetime

# All NHS pages related to CKD
NHS_CKD_URLS = [
    "https://www.nhs.uk/conditions/kidney-disease/",
    "https://www.nhs.uk/conditions/kidney-disease/symptoms/",
    "https://www.nhs.uk/conditions/kidney-disease/diagnosis/",
    "https://www.nhs.uk/conditions/kidney-disease/treatment/",
    "https://www.nhs.uk/conditions/kidney-disease/living-with/",
    "https://www.nhs.uk/conditions/kidney-disease/prevention/",
    # Related conditions patients often ask about
    "https://www.nhs.uk/conditions/kidney-disease/causes/",
    "https://www.nhs.uk/conditions/kidney-failure/",
    "https://www.nhs.uk/conditions/dialysis/",
    "https://www.nhs.uk/conditions/kidney-transplant/",
    "https://www.nhs.uk/conditions/kidney-infections/",
    "https://www.nhs.uk/conditions/kidney-stones/",
    "https://www.nhs.uk/conditions/acute-kidney-injury/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str) -> Optional[str]:
    """Fetch a page and return its HTML, or None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        print(f"  ✓ Fetched: {url}")
        return response.text
    except requests.RequestException as e:
        print(f"  ✗ Failed:  {url} — {e}")
        return None


def clean_text(text: str) -> str:
    """Normalise whitespace and strip junk from extracted text."""
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = text.strip()
    return text


def extract_sections(html: str, url: str) -> List[dict]:
    """
    Parse an NHS page and split it into section-based chunks.

    NHS pages use a consistent structure:
      <article>
        <h2>Section heading</h2>
        <p>Content...</p>
        ...
      </article>

    Each <h2> (or <h3>) block becomes one chunk.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Get the page title
    title_tag = soup.find("title")
    page_title = clean_text(title_tag.get_text()) if title_tag else "Unknown"
    # Remove the " - NHS" suffix that appears on all pages
    page_title = page_title.replace(" - NHS", "").strip()

    # Find the main content area — NHS uses <article> or <main>
    content_area = soup.find("article") or soup.find("main")
    if not content_area:
        print(f"  ⚠ No <article> or <main> found on {url}")
        return []

    chunks = []
    # We'll walk through all heading + content blocks
    # Strategy: find all h2 tags, then grab all sibling content until next h2
    headings = content_area.find_all(["h2", "h3"])

    if not headings:
        # No headings — treat the whole page as one chunk
        full_text = clean_text(content_area.get_text())
        if len(full_text) > 50:  # skip near-empty pages
            chunks.append({
                "page_title": page_title,
                "section_title": page_title,
                "content": full_text,
                "source_url": url,
            })
        return chunks

    # Also grab any intro text BEFORE the first heading
    intro_parts = []
    for elem in content_area.children:
        if elem == headings[0] or (hasattr(elem, "name") and elem.name in ["h2", "h3"]):
            break
        if hasattr(elem, "get_text"):
            t = clean_text(elem.get_text())
            if t:
                intro_parts.append(t)

    if intro_parts:
        intro_text = " ".join(intro_parts)
        if len(intro_text) > 50:
            chunks.append({
                "page_title": page_title,
                "section_title": f"{page_title} — Overview",
                "content": intro_text,
                "source_url": url,
            })

    # Now process each heading and its following content
    for i, heading in enumerate(headings):
        section_title = clean_text(heading.get_text())

        # Collect all content between this heading and the next one
        content_parts = []
        sibling = heading.find_next_sibling()
        while sibling:
            # Stop if we hit the next heading at same or higher level
            if sibling.name in ["h2", "h3"]:
                break
            if hasattr(sibling, "get_text"):
                t = clean_text(sibling.get_text())
                if t:
                    content_parts.append(t)
            sibling = sibling.find_next_sibling()

        section_content = " ".join(content_parts)

        # Only keep chunks with meaningful content
        if len(section_content) > 30:
            chunks.append({
                "page_title": page_title,
                "section_title": section_title,
                "content": section_content,
                "source_url": url,
            })

    return chunks


def make_chunk_id(url: str, section_title: str, index: int) -> str:
    """Create a readable unique ID for each chunk."""
    # Extract the path part: /conditions/kidney-disease/symptoms/ -> kidney-disease-symptoms
    path = url.replace("https://www.nhs.uk/conditions/", "")
    path = path.strip("/").replace("/", "-")
    if not path:
        path = "overview"

    # Clean section title for use in ID
    section_slug = re.sub(r"[^a-z0-9]+", "-", section_title.lower()).strip("-")[:40]

    return f"nhs-{path}-{section_slug}-{index:03d}"


def main():
    print("=" * 60)
    print("NHS CKD Content Scraper")
    print("=" * 60)

    all_chunks = []
    chunk_counter = 0

    for url in NHS_CKD_URLS:
        html = fetch_page(url)
        if not html:
            continue

        sections = extract_sections(html, url)
        for section in sections:
            chunk_counter += 1
            section["chunk_id"] = make_chunk_id(
                url, section["section_title"], chunk_counter
            )
            section["word_count"] = len(section["content"].split())
            all_chunks.append(section)

    # Save to JSON
    output = {
        "metadata": {
            "scraped_at": datetime.now().isoformat(),
            "source": "NHS UK",
            "topic": "Chronic Kidney Disease (CKD)",
            "total_chunks": len(all_chunks),
            "total_words": sum(c["word_count"] for c in all_chunks),
            "urls_scraped": len(NHS_CKD_URLS),
        },
        "chunks": all_chunks,
    }

    output_path = "nhs_ckd_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Done! Saved {len(all_chunks)} chunks to {output_path}")
    print(f"Total words: {output['metadata']['total_words']}")
    print("=" * 60)

    # Print a preview of the first 3 chunks
    print("\n📄 Preview of first 3 chunks:\n")
    for chunk in all_chunks[:3]:
        print(f"  ID:      {chunk['chunk_id']}")
        print(f"  Page:    {chunk['page_title']}")
        print(f"  Section: {chunk['section_title']}")
        print(f"  Words:   {chunk['word_count']}")
        print(f"  Content: {chunk['content'][:120]}...")
        print()


if __name__ == "__main__":
    main()