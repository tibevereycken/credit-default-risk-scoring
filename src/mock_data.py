import numpy as np
import pandas as pd

def generate_synthetic_credit_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Target genereren met ~4.4% default rate
    y = np.random.binomial(n=1, p=0.044, size=n_samples)
    
    # 2. Continue variabelen (financiële ratio's met positieve correlatie met default)
    leverage = np.random.exponential(scale=1.5, size=n_samples) + (y * 1.2)
    liquidity = np.random.gamma(shape=2.0, scale=1.0, size=n_samples) - (y * 0.5)
    liquidity = np.clip(liquidity, 0.01, None)
    turnover = np.random.normal(loc=1.2, scale=0.4, size=n_samples)
    
    # Continue variabelen met veel nullen en uitschieters
    zero_inflated = np.random.exponential(scale=2.0, size=n_samples)
    zero_inflated[np.random.rand(n_samples) > 0.3] = 0
    
    # 3. Categorische variabelen (lage en hoge kardinaliteit)
    sector = np.random.choice(["Retail", "Industry", "Services", "Construction", "Tech"], size=n_samples)
    region_code = np.random.choice([f"REG_{i:02d}" for i in range(1, 50)], size=n_samples)
    
    df = pd.DataFrame({
        "leverage_ratio": leverage,
        "current_ratio": liquidity,
        "asset_turnover": turnover,
        "overdue_days": zero_inflated,
        "sector": sector,
        "region_code": region_code,
        "IN_DEFAULT_FLG": y
    })
    
    # Simuleer ontbrekende waarden
    mask_missing = np.random.rand(*df.shape) < 0.03
    mask_missing[:, -1] = False  # target mag geen missing values hebben
    df = df.mask(mask_missing)
    
    return df