import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype("int32")
    df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype("float32")

    df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
    df["Ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)

    res_split = df["ScreenResolution"].str.split("x", n=1, expand=True)
    df["X_res"] = (
        res_split[0]
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d+)", expand=False)
        .astype("int")
    )
    df["Y_res"] = res_split[1].astype("int")
    df["ppi"] = (((df["X_res"] ** 2) + (df["Y_res"] ** 2)) ** 0.5 / df["Inches"]).astype("float")

    df.drop(columns=["ScreenResolution", "Inches", "X_res", "Y_res"], inplace=True)

    df["Cpu Name"] = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))

    def fetch_processor(text: str) -> str:
        if text in ["Intel Core i7", "Intel Core i5", "Intel Core i3"]:
            return text
        if text.split()[0] == "Intel":
            return "Other Intel Processor"
        return "AMD Processor"

    df["Cpu brand"] = df["Cpu Name"].apply(fetch_processor)
    df.drop(columns=["Cpu", "Cpu Name"], inplace=True)

    # Memory feature engineering from the notebook pipeline.
    mem = df["Memory"].astype(str).replace("\\.0", "", regex=True)
    mem = mem.str.replace("GB", "", regex=False)
    mem = mem.str.replace("TB", "000", regex=False)
    mem_split = mem.str.split("+", n=1, expand=True)

    first = mem_split[0].str.strip()
    second = mem_split[1].fillna("0")

    layer1_hdd = first.apply(lambda x: 1 if "HDD" in x else 0)
    layer1_ssd = first.apply(lambda x: 1 if "SSD" in x else 0)
    layer1_hybrid = first.apply(lambda x: 1 if "Hybrid" in x else 0)
    layer1_flash = first.apply(lambda x: 1 if "Flash Storage" in x else 0)

    layer2_hdd = second.apply(lambda x: 1 if "HDD" in x else 0)
    layer2_ssd = second.apply(lambda x: 1 if "SSD" in x else 0)
    layer2_hybrid = second.apply(lambda x: 1 if "Hybrid" in x else 0)
    layer2_flash = second.apply(lambda x: 1 if "Flash Storage" in x else 0)

    first_num = first.str.extract(r"(\d+)", expand=False).fillna(0).astype(int)
    second_num = second.str.extract(r"(\d+)", expand=False).fillna(0).astype(int)

    df["HDD"] = (first_num * layer1_hdd + second_num * layer2_hdd)
    df["SSD"] = (first_num * layer1_ssd + second_num * layer2_ssd)
    df["Hybrid"] = (first_num * layer1_hybrid + second_num * layer2_hybrid)
    df["Flash_Storage"] = (first_num * layer1_flash + second_num * layer2_flash)

    df.drop(columns=["Memory", "Hybrid", "Flash_Storage"], inplace=True)

    df["Gpu brand"] = df["Gpu"].apply(lambda x: x.split()[0])
    df = df[df["Gpu brand"] != "ARM"].copy()
    df.drop(columns=["Gpu"], inplace=True)

    def cat_os(inp: str) -> str:
        if inp in ["Windows 10", "Windows 7", "Windows 10 S"]:
            return "Windows"
        if inp in ["macOS", "Mac OS X"]:
            return "Mac"
        return "Others/No OS/Linux"

    df["os"] = df["OpSys"].apply(cat_os)
    df.drop(columns=["OpSys"], inplace=True)

    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "laptop_data.csv"
    out_df_path = base_dir / "df.pkl"
    out_pipe_path = base_dir / "pipe.pkl"

    raw_df = pd.read_csv(csv_path)
    df = build_features(raw_df)

    x = df.drop(columns=["Price"])
    y = np.log(df["Price"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    except TypeError:
        # sklearn < 1.2 uses `sparse` instead of `sparse_output`.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, [0, 1, 7, 10, 11]),
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=3,
        max_depth=15,
        max_features=0.75,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(x_train, y_train)

    with open(out_df_path, "wb") as f:
        pickle.dump(df, f)

    with open(out_pipe_path, "wb") as f:
        pickle.dump(pipe, f)

    print("Rebuilt artifacts:")
    print(f"- {out_df_path}")
    print(f"- {out_pipe_path}")


if __name__ == "__main__":
    main()
