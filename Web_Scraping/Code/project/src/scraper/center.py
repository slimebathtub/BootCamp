import pandas as pd
from .fetcher import *
from .pipline import final_combine_parse_hours_table

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
        self.time_data_df = final_combine_parse_hours_table(self.html)
        return self.time_data_df

    # For debugging
    def print_dic(self):
        print(self.time_data_df.to_dict(orient="records") or "No data")

    # For debugging
    def to_csv(self):
        filname = f"{self.name}.csv"
        self.time_data_df.to_csv(filname, index=False)
