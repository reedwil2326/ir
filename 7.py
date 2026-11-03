# Practical 7: Web Crawler - product names, prices and links from an e-commerce site
# Practice site made for scraping: https://books.toscrape.com

import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Step 1: Starting URL (from command line, or the default first page)
url = sys.argv[1] if len(sys.argv) > 1 else "https://books.toscrape.com/catalogue/page-1.html"
products = []
pages = 3                                                  # crawl the first 3 pages

# Step 2: Crawl page by page
for _ in range(pages):
    # FIX: .content instead of .text, so £ is not shown as Â£
    soup = BeautifulSoup(requests.get(url, timeout=10).content, "html.parser")

    # Step 3: Extract every product card on the page
    for item in soup.select("article.product_pod"):
        a = item.h3.a
        products.append({
            "title": a["title"],                           # full title (link text is shortened)
            "price": item.select_one(".price_color").text,
            "link":  urljoin(url, a["href"]),              # relative link -> full link
        })

    # Step 4: Follow the "next" button, stop if there is none
    nxt = soup.select_one("li.next a")
    if not nxt:
        break
    url = urljoin(url, nxt["href"])

# Step 5: Save to CSV and show a preview
df = pd.DataFrame(products)
df.to_csv("products.csv", index=False)
print(df.head(), f"\nTotal products: {len(df)}  (saved to products.csv)")