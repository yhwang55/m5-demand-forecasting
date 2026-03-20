import pandas as pd

def add_time_features(df, date_col="date"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day"] = df[date_col].dt.day
    df["dow"] = df[date_col].dt.dayofweek
    return df