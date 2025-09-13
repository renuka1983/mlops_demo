import pandas as pd

def test_data_schema():
    df = pd.read_csv("data.csv")
    expected_cols = {"feature1", "feature2", "label"}
    assert expected_cols.issubset(df.columns)
