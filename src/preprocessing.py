import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def compute_woe(df: pd.DataFrame, feature: str, target: str, smooth: float = 0.5) -> dict:
    """Berekent Weight of Evidence (WoE) mapping voor een categorische kolom."""
    total_pos = (df[target] == 1).sum()
    total_neg = (df[target] == 0).sum()
    
    woe_dict = {}
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    
    for cat, row in grouped.iterrows():
        pos = row['sum'] + smooth
        neg = (row['count'] - row['sum']) + smooth
        dist_pos = pos / total_pos
        dist_neg = neg / total_neg
        woe_dict[cat] = np.log(dist_pos / dist_neg)
        
    return woe_dict

def preprocess_credit_pipeline(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):
    """Schoont en transformeert features conform credit-risk standaarden."""
    num_cols = ["leverage_ratio", "current_ratio", "asset_turnover", "overdue_days"]
    cat_low_card = ["sector"]
    cat_high_card = ["region_code"]
    
    # 1. Missing values opvullen: mediaan voor numeriek, modus voor categorisch
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
        
    for col in cat_low_card + cat_high_card:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)
        
    # 2. WoE encoding voor high-cardinality feature
    train_full = X_train.copy()
    train_full["target"] = y_train.values
    woe_map = compute_woe(train_full, "region_code", "target")
    
    X_train["region_code_woe"] = X_train["region_code"].map(woe_map).fillna(0)
    X_test["region_code_woe"] = X_test["region_code"].map(woe_map).fillna(0)
    
    # 3. RobustScaler op continue features (ongevoelig voor uitschieters)
    scaler = RobustScaler()
    scaled_train_num = scaler.fit_transform(X_train[num_cols + ["region_code_woe"]])
    scaled_test_num = scaler.transform(X_test[num_cols + ["region_code_woe"]])
    
    # 4. One-Hot Encoding voor lage kardinaliteit
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_train = ohe.fit_transform(X_train[cat_low_card])
    ohe_test = ohe.transform(X_test[cat_low_card])
    
    # Samenvoegen
    X_train_proc = np.hstack([scaled_train_num, ohe_train])
    X_test_proc = np.hstack([scaled_test_num, ohe_test])
    
    return X_train_proc, X_test_proc