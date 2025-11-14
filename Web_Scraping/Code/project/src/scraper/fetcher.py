# ---------- fetcher ----------
import random
import time
import requests

UA = "CS-Club-HoursScraper/1.0 (+contact: you@example.edu)"
session = requests.Session()
session.headers.update({"User-Agent": UA})

def fetcher(url, rps=0.6, timeout=20):
    time.sleep(max(1.0 / rps, 0.2) + random.uniform(0, 0.3))
    r = session.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    print(f"Fetched {url} with status {r.status_code}")
    return r.text