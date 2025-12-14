from metapub import FindIt
import requests

pmids = [2020202, 1745076, 2768771, 8277124, 4031339]

for pmid in pmids:
    src = FindIt(pmid)

    # src.pma contains the PubMedArticle
    print(src.pma.title)
    print(src.pma.abstract)

    # URL, if available, will be fulltext PDF
    if src.url:
        # insert your downloader of choice here, e.g. requests.get(url)
    else:
        # if no URL, reason is one of "PAYWALL", "TXERROR", or "NOFORMAT"
       print(src.reason)