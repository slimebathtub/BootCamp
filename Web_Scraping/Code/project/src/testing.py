# %% ==== 小工具：通用的檢查器 ====
from parese import extract_day_time_rows, find_hours_table, rows_to_dataframe


def _check(label, condition, expected=None, got=None):
    if condition:
        print(f"  ✅ {label}")
        return 1, 1
    else:
        print(f"  🔴 {label}")
        if expected is not None or got is not None:
            print("     expected:", expected)
            print("     got     :", got)
        return 0, 1


# %% ==== 1) find_hours_table：safely smoke ====
def smoke_find_hours_table_safe():
    print("=== Smoke: find_hours_table ===")
    html_precise = """
    <html><body>
      <table class="table table-striped table-csm"><tbody><tr><td>A</td><td>B</td></tr></tbody></table>
    </body></html>
    """
    html_fallback = """
    <html><body>
      <table><tbody><tr><td>X</td><td>Y</td></tr></tbody></table>
    </body></html>
    """
    html_none = "<div>No table here</div>"

    passed = total = 0

    try:
        t1 = find_hours_table(html_precise)
        p, t = _check("precise selector finds table", t1 is not None, True, t1 is not None)
        passed += p; total += t

        t2 = find_hours_table(html_fallback)
        p, t = _check("fallback finds table", t2 is not None, True, t2 is not None)
        passed += p; total += t

        t3 = find_hours_table(html_none)
        p, t = _check("no table returns None", t3 is None, True, t3 is None)
        passed += p; total += t
    except Exception as e:
        print("  🔴 Exception:", e)

    print(f"-- Result: {passed}/{total} passed\n")


# %% ==== 2) extract_day_time_rows：safely smoke ====
def _mk_table(inner, cls='table table-striped table-csm'):
    return f'<table class="{cls}"><tbody>{inner}</tbody></table>'

def smoke_extract_day_time_rows_safe():
    print("=== Smoke: extract_day_time_rows ===")
    passed = total = 0

    try:
        # Case 1: need expand (mon-thu)
        html_range = _mk_table("""
          <tr><td>Mon-Thu</td><td>8:00 am - 5:00 pm</td></tr>
          <tr><td>Fri</td><td>8:00 am - 1:00 pm</td></tr>
        """)
        table_range = find_hours_table(html_range)
        rows_range = extract_day_time_rows(table_range)
        print("[Case 1 output]:", rows_range)

        cond1 = any(r["Day"] == "Monday" for r in rows_range)
        p, t = _check("range contains Monday", cond1, True, cond1); passed+=p; total+=t
        cond2 = any(r["Day"] == "Thursday" for r in rows_range)
        p, t = _check("range contains Thursday", cond2, True, cond2); passed+=p; total+=t
        cond3 = any(r["Day"] == "Friday" for r in rows_range)
        p, t = _check("range contains Friday", cond3, True, cond3); passed+=p; total+=t

        # Case 2: no expand（copmltie day name）
        html_full = _mk_table("""
          <tr><td>Monday</td><td>9:00 am - 4:00 pm</td></tr>
          <tr><td>Friday</td><td>10:00 am - 2:00 pm</td></tr>
        """)
        table_full = find_hours_table(html_full)
        rows_full = extract_day_time_rows(table_full)
        print("[Case 2 output]:", rows_full)

        expected_full = [
            {'Day': 'Monday', 'Time': '9:00 am - 4:00 pm'},
            {'Day': 'Friday', 'Time': '10:00 am - 2:00 pm'}
        ]
        p, t = _check("full-name rows match",
                      rows_full == expected_full, expected_full, rows_full)
        passed+=p; total+=t

        # Case 3: ignore the extra columns
        html_extra_cols = _mk_table("""
          <tr><td>Tue</td><td>9-4</td><td>note</td></tr>
        """)
        table_extra = find_hours_table(html_extra_cols)
        rows_extra = extract_day_time_rows(table_extra)
        print("[Case 3 output]:", rows_extra)

        expected_extra = [{'Day':'Tuesday','Time':'9-4'}]
        p, t = _check("extra column ignored",
                      rows_extra == expected_extra, expected_extra, rows_extra)
        passed+=p; total+=t

    except Exception as e:
        print("  🔴 Exception:", e)

    print(f"-- Result: {passed}/{total} passed\n")


#%% ==== 3) rows_to_dataframe: safely smoke ====
def smoke_rows_to_dataframe_safe():
    print("=== Smoke: rows_to_dataframe ===")
    passed = total = 0
    try:
        rows = [
            {"Day":"Monday","Time":"8-5"},
            {"Day":"Tuesday","Time":"8-5"},
            {"Day":"Tuesday","Time":"9-4"},  # duplicate day
        ]
        df = rows_to_dataframe(rows)
        print("[df]:\n", df)

        # delete repeate Tuesday=8-5
        cond_dup = (df[df["Day"]=="Tuesday"]["Time"].iloc[0] == "8-5") and (df["Day"].value_counts().get("Tuesday",0)==1)
        p, t = _check("deduplicate keeps first Tuesday=8-5 and single entry",
                      cond_dup, True, cond_dup)
        passed+=p; total+=t

        empty_df = rows_to_dataframe([])
        print("[empty df shape]:", empty_df.shape)
        p, t = _check("empty rows -> empty df with 2 columns",
                      empty_df.shape == (0,2), (0,2), empty_df.shape)
        passed+=p; total+=t

    except Exception as e:
        print("  🔴 Exception:", e)

    print(f"-- Result: {passed}/{total} passed\n")


#%% ==== testing ====
smoke_find_hours_table_safe()
smoke_extract_day_time_rows_safe()
smoke_rows_to_dataframe_safe()
