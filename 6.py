# Practical 6: PageRank Algorithm (Power Iteration) - verified with networkx

import numpy as np
import networkx as nx

# Step 1: Web graph -> page : pages it links to
links = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C", "A"],
    "E": ["D", "C"],
}
pages = sorted(links)
n, d = len(pages), 0.85                          # n = number of pages, d = damping factor
idx = {p: i for i, p in enumerate(pages)}        # page name -> row/column number

# Step 2: Build the link matrix M
# M[j][i] = 1/(outlinks of i) if page i links to page j
M = np.zeros((n, n))
for p, outs in links.items():
    for o in outs:
        M[idx[o], idx[p]] = 1 / len(outs)

# Step 3: Power iteration -> PR = (1-d)/n + d * M * PR
rank = np.ones(n) / n                            # start with equal rank
for _ in range(100):
    rank = (1 - d) / n + d * M @ rank

# Step 4: Print pages from highest to lowest rank
for p in sorted(pages, key=lambda p: -rank[idx[p]]):
    print(f"{p}: {rank[idx[p]]:.4f}")

# Step 5: Verify with the library version
G = nx.DiGraph([(p, o) for p, outs in links.items() for o in outs])
print("networkx:", {k: round(v, 4) for k, v in nx.pagerank(G, alpha=d).items()})

# For real web pages: use requests + BeautifulSoup to collect each page's <a href> links
# into the `links` dictionary above (see p7_web_crawler.py), then run the same code.