# %% import
from typing import Optional
from bs4 import BeautifulSoup
import pandas as pd
from pyparsing import Tag
import requests
import time
import random
# %%　constants
WEEK = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEK_SYMNBOL = {
    "Monday":"Mon",
    "Tuesday":"Tue",
    "Wednesday":"Wed",
    "Thursday":"Thu",
    "Friday":"Fri",
    "Saturday":"Sat",
    "Sunday":"Sun"
}
DAY_MAP = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
    "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"
}

# %%　preprocess funciton: expand the day range
def expand_day_range(day_text: str) -> list[str]:
    """
    e.g. "Mon-Thu" -> ["Monday","Tuesday","Wednesday","Thursday"]
         "Fri"     -> ["Friday"]
    """
    parts = day_text.split("-")
    if len(parts) == 1:
        return [DAY_MAP.get(parts[0], parts[0])]   # 單一天
    if len(parts) == 2:
        start = DAY_MAP.get(parts[0], parts[0])
        end   = DAY_MAP.get(parts[1], parts[1])
        if start in WEEK and end in WEEK:
            s, e = WEEK.index(start), WEEK.index(end)
            return WEEK[s:e+1]
    return []

# %% fetcher
# ---------- fetcher ----------
UA = "CS-Club-HoursScraper/1.0 (+contact: you@example.edu)"
session = requests.Session()
session.headers.update({"User-Agent": UA})

def fetcher(url, rps=0.6, timeout=20):
    time.sleep(max(1.0 / rps, 0.2) + random.uniform(0, 0.3))
    r = session.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    print(f"Fetched {url} with status {r.status_code}")
    return r.text

# %% Center class
# ---------- Center ----------
class Center:
    def __init__(self, id, name, url):
        self.id = id
        self.name = name
        self.url = url
        self.html = ""
        self.time_data_df = pd.DataFrame(columns=["Day", "Time"]) # default empty

    def fetch_time_data(self):
        self.html = fetcher(self.url)
        self.time_data_df = parse_hours_table(self.html)
        return self.time_data_df

    # For debugging
    def print_dic(self):
        print(self.time_data_df.to_dict(orient="records") or "No data")

    # For debugging
    def to_csv(self):
        filname = f"{self.name}.csv"
        self.time_data_df.to_csv(filname, index=False)

# %% fetch the table from the HTML
# ---------- (TODO) HTML → table ----------
def find_hours_table(html: str)-> Optional[Tag]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.table.table-striped.table-csm")
    if table is None:
        table = soup.find("table")
    return table

# %% fetch rows from the table
# ---------- (TODO) table → rows ----------
def extract_day_time_rows(table) -> list[dict]:
    rows = []
    if table is None:
        return rows
    
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        day_text = tds[0].get_text(strip=True)
        time_text = tds[1].get_text(" ", strip=True)

        # --- don't change here ---
        # expand the day range
        expanded_days = expand_day_range(day_text)
        for d in expanded_days:
            rows.append({"Day": d, "Time": time_text})

    return rows

# %% convert rows to DataFrame
# ---------- (TODO) rows → DataFrame----------
def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Day", "Time"])
    df = pd.DataFrame(rows, columns=["Day", "Time"])

    # delete the repetitive days if any
    df = df.drop_duplicates(subset=["Day"], keep="first")
    return df

# %% complete the DataFrame
# ---------- filled the lost day and frame ----------
def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = pd.DataFrame({"Day": WEEK, "Time": ["Closed"] * 7})
    else:
        existing = set(df["Day"])
        missing = [{"Day": d, "Time": "Closed"} for d in WEEK if d not in existing]
        if missing:
            df = pd.concat([df, pd.DataFrame(missing)], ignore_index=True)
        df["__ord"] = df["Day"].apply(lambda d: WEEK.index(d))
        df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    
    # declare the type to string
    df["Day"] = df["Day"].astype(str)
    df["Time"] = df["Time"].astype(str)
    return df

# %% main final function
# ---------- HTML → DataFrame ----------
def parse_hours_table(html: str) -> pd.DataFrame:
    table = find_hours_table(html)
    rows = extract_day_time_rows(table)
    df = rows_to_dataframe(rows)
    df = finalize_dataframe(df)
    return df
