# filepath: tutorials/python_basics_to_ml.py
"""
Python 新手 Tutorial：從零到基礎 ML/AI
執行方式：
    python tutorials/python_basics_to_ml.py           # 顯示選單
    python tutorials/python_basics_to_ml.py --all     # 一次跑全部
依需求可用參數執行各一節，見底部 main()。
"""

from __future__ import annotations

import sys
import argparse
from typing import Any, Iterable, Callable, Optional

# ------ 基礎：Hello World ------

def hello_world() -> None:
    """示範最小可執行程式與輸出。"""
    print("Hello, World!")  # 為了經典：大小寫與標點一致
    # 挑戰：把 "Hello, World!" 改造成 "Hello, <你的名字>!"
    # 提示：用 f-string，例如 name = "Alice"; print(f"Hello, {name}!")
    name = "Python Beginner"
    print(f"Hello, {name}!")


# ------ 匯入模組 ------

def import_demo() -> None:
    """展示 import 與 from-import、模組屬性檢查。"""
    import math
    from statistics import mean

    nums = [3, 1, 4, 1, 5]
    print("pi ≈", math.pi)
    print("mean(nums) =", mean(nums))
    # 挑戰：使用 math.isclose 驗證圓周率近似
    print("isclose(pi, 3.14, rel_tol=1e-3) ->", math.isclose(math.pi, 3.14, rel_tol=1e-3))


# ------ 變數/型別/格式化 ------

def variables_and_types() -> None:
    """介紹常見型別與 f-string。"""
    an_int: int = 42
    a_float: float = 3.1415
    a_bool: bool = True
    a_str: str = "hello"
    a_list: list[int] = [1, 2, 3]
    a_tuple: tuple[int, int] = (4, 5)
    a_set: set[int] = {1, 2, 2, 3}
    a_dict: dict[str, int] = {"a": 1, "b": 2}

    print(f"{an_int=}, {a_float=:.2f}, {a_bool=}, {a_str=}")
    print(f"{a_list=}, {a_tuple=}, {a_set=}, {a_dict=}")
    # 挑戰：把 a_list 全部平方（用 list comprehension）
    squared = [x * x for x in a_list]
    print("squared:", squared)


# ------ 流程控制：if / for / while ------

def control_flow_demo() -> None:
    """示範 if/elif/else, for, while, break/continue。"""
    x = 7
    if x % 2 == 0:
        print("x is even")
    elif x % 3 == 0:
        print("x is divisible by 3")
    else:
        print("x is odd and not divisible by 3")

    print("for over range:")
    for i in range(5):
        if i == 2:
            continue  # 為了示範：略過 2
        print(i, end=" ")
    print()

    print("while countdown:")
    n = 5
    while n > 0:
        print(n, end=" ")
        if n == 3:
            # 挑戰：試著把 break 改成 pass 看差別
            pass  # 為何：示範語法存在但不做事
        n -= 1
    print()


# ------ 函式語法與型別註解 ------

def sum_all(*nums: float) -> float:
    """回傳所有參數總和。為示範可變參數。"""
    return sum(nums)

def power(base: float, exp: float = 2) -> float:
    """預設 exp=2，示範預設參數。"""
    return base ** exp

def apply(func: Callable[[float], float], values: Iterable[float]) -> list[float]:
    """把函式套用到可迭代資料。"""
    return [func(v) for v in values]

def function_demo() -> None:
    """示範函式、多種參數、lambda 與型別註解。"""
    print("sum_all(1,2,3) =", sum_all(1, 2, 3))
    print("power(3) =", power(3))
    print("power(2, exp=3) =", power(2, exp=3))
    doubled = apply(lambda v: v * 2, [1, 2, 3])
    print("apply(lambda*2, [1,2,3]) ->", doubled)
    # 挑戰：寫個函式 rev(s: str) -> str 回傳字串反轉，並測試 "abc" -> "cba"
    def rev(s: str) -> str:
        return s[::-1]
    assert rev("abc") == "cba"
    print("rev('abc') ok")


# ------ 集合操作與推導式 ------

def collections_demo() -> None:
    """list/dict 操作與 comprehension。"""
    words = ["apple", "banana", "cherry"]
    lengths = [len(w) for w in words]
    print("lengths:", lengths)

    freq = {ch: ord(ch) for ch in "abc"}  # 映射示例
    print("freq:", freq)
    # 挑戰：把 words 轉成 {word: len(word)} 的 dict
    word_len = {w: len(w) for w in words}
    print("word_len:", word_len)


# ------ 例外處理 ------

class PositiveNumberError(ValueError):
    """當期望正數卻收到非正數時拋出。為教學自訂例外。"""

def sqrt_safe(x: float) -> float:
    """對 x<0 拋出 PositiveNumberError。"""
    if x < 0:
        raise PositiveNumberError("x must be non-negative")
    return x ** 0.5

def exceptions_demo() -> None:
    """try/except/else/finally 範例。"""
    for v in [4, -1]:
        try:
            r = sqrt_safe(v)
        except PositiveNumberError as e:
            print("caught:", e)
        else:
            print("sqrt:", r)
        finally:
            pass  # 為何：示範 finally 仍會執行


# ------ 檔案 I/O ------

def file_io_demo(path: str = "demo.txt") -> None:
    """建立文字檔並讀回。"""
    content = "first line\nsecond line\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(path, "r", encoding="utf-8") as f:
        back = f.read()
    print("file content:\n", back)
    # 挑戰：把每行前面加上行號再輸出
    numbered = "".join(f"{i+1}: {line}" for i, line in enumerate(back.splitlines(True)))
    print("numbered:\n", numbered)


# ------ Numpy / Pandas (資料處理入門) ------

def numpy_pandas_demo() -> None:
    """若套件存在：示範陣列與資料框基本操作；否則給出安裝提示。"""
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:
        print("[提示] 需要 numpy 與 pandas。安裝：pip install numpy pandas")
        print("錯誤訊息：", e)
        return

    a = np.array([1, 2, 3, 4], dtype=float)
    b = np.linspace(0, 1, 5)
    print("a+b[:4] ->", a + b[:4])
    print("a.mean() ->", a.mean())

    df = pd.DataFrame({"x": a, "y": a ** 2})
    print("DataFrame head:\n", df.head())
    print("describe:\n", df.describe())

    # 挑戰：新增一欄 z = x*y 並顯示前三列
    df["z"] = df["x"] * df["y"]
    print("df with z (head):\n", df.head(3))


# ------ Matplotlib (繪圖入門) ------

def matplotlib_demo() -> None:
    """產生一張簡單圖。無圖形環境時只提示成功/失敗。"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[提示] 需要 matplotlib。安裝：pip install matplotlib")
        print("錯誤訊息：", e)
        return

    x = np.arange(0, 10, 1)
    y = x ** 2
    try:
        plt.figure()
        plt.plot(x, y)  # why：最少即用，避免過度樣式
        plt.title("y = x^2")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig("plot.png")
        print("已輸出圖檔：plot.png")
        plt.close()
    except Exception as e:
        print("[提示] 繪圖可能需要圖形後端或寫權限：", e)


# ------ 最小可行 ML：分類（Iris） ------

def ml_minimum_viable_demo() -> None:
    """使用 sklearn 進行最小可行分類範例，並比較兩個基線模型。"""
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
    except Exception as e:
        print("[提示] 需要 scikit-learn。安裝：pip install scikit-learn")
        print("錯誤訊息：", e)
        return

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Pipeline：標準化 + 模型
    knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    lr_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

    knn_clf.fit(X_train, y_train)
    lr_clf.fit(X_train, y_train)

    knn_acc = accuracy_score(y_test, knn_clf.predict(X_test))
    lr_acc = accuracy_score(y_test, lr_clf.predict(X_test))

    print(f"KNN accuracy: {knn_acc:.3f}")
    print(f"LogReg accuracy: {lr_acc:.3f}")
    better = "KNN" if knn_acc >= lr_acc else "LogReg"
    print(f"Winner (on this split): {better}")

    # 挑戰：用 cross_val_score 比較更穩定；或替換為 SVC / RandomForest


# ------ 一鍵跑全部 ------

def run_all() -> None:
    """依序跑過所有章節。"""
    hello_world()
    import_demo()
    variables_and_types()
    control_flow_demo()
    function_demo()
    collections_demo()
    exceptions_demo()
    file_io_demo()
    numpy_pandas_demo()
    matplotlib_demo()
    ml_minimum_viable_demo()


# ------ CLI 入口 ------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python 新手 Tutorial：從零到基礎 ML/AI")
    p.add_argument("--all", action="store_true", help="一次跑全部章節")
    p.add_argument("--sec", type=str, default="", help="跑單一章節 (e.g. hello, import, types, control, func, col, ex, io, np_pd, plt, ml)")
    return p

def dispatch_section(name: str) -> bool:
    """回傳 True 表示已匹配到並執行。"""
    table: dict[str, Callable[[], None]] = {
        "hello": hello_world,
        "import": import_demo,
        "types": variables_and_types,
        "control": control_flow_demo,
        "func": function_demo,
        "col": collections_demo,
        "ex": exceptions_demo,
        "io": file_io_demo,
        "np_pd": numpy_pandas_demo,
        "plt": matplotlib_demo,
        "ml": ml_minimum_viable_demo,
    }
    func = table.get(name.strip().lower())
    if func:
        func()
        return True
    return False

def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if args.all:
        run_all()
        return
    if args.sec:
        if not dispatch_section(args.sec):
            print("未知章節。可用：", " | ".join(["hello","import","types","control","func","col","ex","io","np_pd","plt","ml"]))
        return
    # 互動式簡單選單
    print("Python 新手 Tutorial：選擇要執行的章節")
    print("1) hello   2) import   3) types   4) control   5) func")
    print("6) col     7) ex       8) io      9) np_pd     10) plt   11) ml")
    try:
        choice = input("輸入數字（或 q 離開，a 全跑）：").strip().lower()
        if choice == "q":
            return
        if choice == "a":
            run_all()
            return
        idx = int(choice)
    except Exception:
        print("輸入無效。")
        return

    mapping = {
        1: "hello", 2: "import", 3: "types", 4: "control", 5: "func",
        6: "col", 7: "ex", 8: "io", 9: "np_pd", 10: "plt", 11: "ml",
    }
    sec = mapping.get(idx, "")
    if not sec or not dispatch_section(sec):
        print("選擇無效。")

if __name__ == "__main__":
    main()
