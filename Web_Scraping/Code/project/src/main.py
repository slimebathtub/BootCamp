from scraper.parese import extract_day_time_rows, find_hours_table, rows_to_dataframe
from scraper.center import Center

urls = [
    "https://collegeofsanmateo.edu/mrc"
    "https://collegeofsanmateo.edu/learningcenter"
    "https://collegeofsanmateo.edu/labs/isc.asp"
    "https://collegeofsanmateo.edu/esl/eslcenter.php"
    "https://collegeofsanmateo.edu/writing"
]

Centers = [
    Center(1, "MRC", "https://collegeofsanmateo.edu/mrc"),
    Center(2, "Learning Center", "https://collegeofsanmateo.edu/learningcenter"),
    Center(3, "ISC", "https://collegeofsanmateo.edu/labs/isc.asp"),
    Center(4, "ESL Center", "https://collegeofsanmateo.edu/esl/eslcenter.php"),
    Center(5, "Writing Center", "https://collegeofsanmateo.edu/writing"),
]

if __name__ == "__main__":
    for center in Centers:
        print(f"--- Fetching data for {center.name} ---")
        df = center.fetch_time_data()
        print(df)
        print()
    print("=== All centers data fetched ===")
