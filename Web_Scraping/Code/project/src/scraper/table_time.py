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
        return [DAY_MAP.get(parts[0], parts[0])]   # only one day
    if len(parts) == 2:
        start = DAY_MAP.get(parts[0], parts[0])
        end   = DAY_MAP.get(parts[1], parts[1])
        if start in WEEK and end in WEEK:
            s, e = WEEK.index(start), WEEK.index(end)
            return WEEK[s:e+1]
    return []


