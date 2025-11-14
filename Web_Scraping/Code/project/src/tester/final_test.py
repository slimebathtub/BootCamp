# %%
from scraper.parese import extract_day_time_rows, finalize_dataframe, find_hours_table, rows_to_dataframe


# %% smoke test for all three TODOs on a given HTML
def smoke_all_three_on_given_html_safe():
    print("=== Smoke Test (All 3 TODOs + e2e) on given HTML ===")

    given_html = """
    <table class="table table-striped table-csm">
      <thead><tr><th>Day</th><th>Time</th></tr></thead>
      <tbody>
        <tr><td>Monday</td><td><span>8:00 am - 4:00 pm</span></td></tr>
        <tr><td>Tuesday</td><td>8:00 am - 4:00 pm</td></tr>
        <tr><td>Wednesday</td><td>8:00 am - 4:00 pm</td></tr>
        <tr><td>Thursday</td><td>8:00 am - 4:00 pm</td></tr>
        <tr><td>Friday</td><td>8:00 am - 2:00 pm</td></tr>
      </tbody>
    </table>
    """

    # --- 1) find_hours_table ---
    try:
        table = find_hours_table(given_html)
        print("[1] find_hours_table ->", "Found" if table is not None else "Not found")
        if table is None:
            print("  🔴 Error: Table not found, expected a <table> element with class.")
        else:
            print("  ✅ Passed\n")
    except Exception as e:
        print("  🔴 Exception in find_hours_table:", e, "\n")

    # --- 2) extract_day_time_rows ---
    try:
        table = find_hours_table(given_html)
        rows = extract_day_time_rows(table)
        print("[2] extract_day_time_rows ->")
        print("  output:", rows)
        expected_rows = [
            {"Day":"Monday",    "Time":"8:00 am - 4:00 pm"},
            {"Day":"Tuesday",   "Time":"8:00 am - 4:00 pm"},
            {"Day":"Wednesday", "Time":"8:00 am - 4:00 pm"},
            {"Day":"Thursday",  "Time":"8:00 am - 4:00 pm"},
            {"Day":"Friday",    "Time":"8:00 am - 2:00 pm"},
        ]
        if rows == expected_rows:
            print("  ✅ Passed\n")
        else:
            print("  🔴 Mismatch! Expected:", expected_rows, "\n   Got:", rows, "\n")
    except Exception as e:
        print("  🔴 Exception in extract_day_time_rows:", e, "\n")

    # --- 3) rows_to_dataframe + finalize_dataframe ---
    try:
        df = rows_to_dataframe(rows)
        df_final = finalize_dataframe(df)
        print("[3] rows_to_dataframe + finalize_dataframe ->")
        print(df_final)
        expected_records = [
            {"Day":"Monday",   "Time":"8:00 am - 4:00 pm"},
            {"Day":"Tuesday",  "Time":"8:00 am - 4:00 pm"},
            {"Day":"Wednesday","Time":"8:00 am - 4:00 pm"},
            {"Day":"Thursday", "Time":"8:00 am - 4:00 pm"},
            {"Day":"Friday",   "Time":"8:00 am - 2:00 pm"},
            {"Day":"Saturday", "Time":"Closed"},
            {"Day":"Sunday",   "Time":"Closed"},
        ]
        if df_final.to_dict(orient="records") == expected_records:
            print("  ✅ Passed\n")
        else:
            print("  🔴 Mismatch! Expected:", expected_records, "\n   Got:", df_final.to_dict(orient='records'), "\n")
    except Exception as e:
        print("  🔴 Exception in finalize_dataframe:", e, "\n")

    print("=== Smoke Test Completed ===\n")
    
smoke_all_three_on_given_html_safe()

# %%
