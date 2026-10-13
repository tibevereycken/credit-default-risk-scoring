from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_classifiers(X_train, y_train, scale_pos_weight: float):
    """Traint baseline- en geavanceerde ensemble-modellen voor default predictie."""
    models = {
        "Decision_Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=150, learning_rate=0.05, is_unbalance=True, random_state=42, verbose=-1),
        "XGBoost_Weighted": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, 
                                          scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
    }
    
    trained_models = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
    return trained_models