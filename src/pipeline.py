from .data import load_sample_sales
from .features import add_time_features

def build_sample_dataset():
    df = load_sample_sales()
    df = add_time_features(df, date_col="date")
    return df