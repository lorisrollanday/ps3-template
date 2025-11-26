import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_macro_data():
    """Load main macro data (inflation, gdp growth, unemployment) in tidy panel form."""
    df = pd.read_csv(DATA_DIR / "Infl-Gdp-une....csv")  # put exact name
    # Make sure you have columns: Country, Year, gdp_growth, inflation, unemployment
    return df

def load_govexp_data():
    """Load government expenditure as % of GDP and tidy it."""
    df = pd.read_csv(DATA_DIR / "GovExp%.csv")
    # Clean so you also end up with: Country, Year, govexp_share
    return df

def build_panel():
    """Merge all indicators into one panel dataset."""
    macro = load_macro_data()
    gov   = load_govexp_data()

    # Merge on Country + Year (adapt column names if needed)
    df = macro.merge(
        gov,
        on=["Country Name", "year"],   # or Country / Year depending on your file
        how="inner"
    )

    # Sort
    df = df.sort_values(["Country Name", "year"])

    return df

def make_train_test(df, train_end_year=2015):
    """
    Create features X and target y where target is next-year GDP growth.
    Train on <= train_end_year, test on > train_end_year.
    """
    # Target = next-year GDP growth per country
    df = df.copy()
    df["gdp_growth_t1"] = df.groupby("Country Name")["gdp_growth"].shift(-1)

    # Drop last year of each country where we don't know next year
    df = df.dropna(subset=["gdp_growth_t1"])

    feature_cols = ["gdp_growth", "inflation", "unemployment", "govexp_share"]
    target_col = "gdp_growth_t1"

    train_mask = df["year"] <= train_end_year
    test_mask  = df["year"] > train_end_year

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    return X_train, X_test, y_train, y_test
