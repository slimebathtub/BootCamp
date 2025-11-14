# %% complete the DataFrame
# ---------- filled the lost day and frame ----------
import pandas as pd
from .table_time import WEEK
from .parese import find_hours_table, rows_to_dataframe, extract_day_time_rows

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
def final_combine_parse_hours_table(html: str) -> pd.DataFrame:
    table = find_hours_table(html)
    rows = extract_day_time_rows(table)
    df = rows_to_dataframe(rows)
    df = finalize_dataframe(df)
    return df