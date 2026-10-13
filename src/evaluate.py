import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score

def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    records = []

    for name, clf in models.items():
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        rec = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        records.append({
            "Model": name,
            "ROC_AUC": round(auc, 4),
            "Recall (Sensitivity)": round(rec, 4),
            "Precision": round(prec, 4),
            "False_Negative_Rate": round(fnr, 4)
        })

    return pd.DataFrame(records)