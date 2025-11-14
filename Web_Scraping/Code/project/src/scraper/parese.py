# %% import
from bs4 import BeautifulSoup
import pandas as pd
from pyparsing import Optional, Tag
import requests
import time
import random
from .table_time import WEEK

# %% fetch the table from the HTML
# input: html string, output: the table element
# ---------- (TODO) HTML → table ----------
def find_hours_table(html: str):
    """
    Return the <table> element that contains the hours, or None.
    Strategy ideas:
      - soup.find('table', {'id': 'hours'}) or by class like 'hours'/'schedule'
      - fallback: look for a table whose first column header contains 'Day'
    """
    return None

# %% fetch rows from the table
# ---------- (TODO) table → rows ----------
def extract_day_time_rows(table) -> list[dict]:
    """
    Return list of {'Day': <English day>, 'Time': <text>}.
    Strategy ideas:
      - Iterate <tr>, read first two columns (day, time).
      - If day text is a range like 'Mon-Thu', expand to multiple rows.
      - Expand function: expand_day_range()
    """
    return []

# %% convert rows to DataFrame
# ---------- (TODO) rows → DataFrame----------
def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Convert rows to a DataFrame with columns ['Day','Time'].
    Remove duplicates and keep the first occurrence, then sort by WEEK order.
    """
    # TODO: implement
    return pd.DataFrame(columns=['Day', 'Time'])
