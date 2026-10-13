from sklearn.model_selection import train_test_split
from src.mock_data import generate_synthetic_credit_data
from src.preprocessing import preprocess_credit_pipeline
from src.models import train_classifiers
from src.evaluate import evaluate_models

def run_pipeline():
    print("=" * 60)
    print("CRELAN CREDIT DEFAULT RISK SCORING PIPELINE")
    print("=" * 60)
    
    # 1. Dataset genereren
    df = generate_synthetic_credit_data(n_samples=6000, seed=42)
    X = df.drop(columns=["IN_DEFAULT_FLG"])
    y = df["IN_DEFAULT_FLG"]
    
    default_rate = y.mean() * 100
    print(f"\n[1] Gegevens geladen: {len(df)} observaties")
    print(f"    Target Default Rate: {default_rate:.2f}% (Extreme Class Imbalance)")
    
    # 2. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # 3. Preprocessing (Imputatie, WoE, RobustScaler, OneHot)
    print("\n[2] Feature engineering & Weight of Evidence encoding...")
    X_train_proc, X_test_proc = preprocess_credit_pipeline(X_train, y_train, X_test)
    
    # Bereken scale_pos_weight
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 4. Modellen trainen
    print("\n[3] Modellen trainen (Decision Tree, Random Forest, LightGBM, XGBoost)...")
    models = train_classifiers(X_train_proc, y_train, scale_pos_weight=scale_weight)
    
    # 5. Evaluatie
    print("\n[4] Evaluatie op ongeziene testset:")
    results_df = evaluate_models(models, X_test_proc, y_test)
    print(results_df.to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()