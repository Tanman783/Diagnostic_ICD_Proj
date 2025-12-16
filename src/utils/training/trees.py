from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

def get_xgboost_model():
    """Returns an untrained XGBoost classifier with standard params."""
    return XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="logloss", n_jobs=-1, random_state=42
    )
def get_rf_model():
    """Returns an untrained Random Forest classifier with standard params."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
    )

def get_catboost_model():
    """Returns an untrained CatBoost classifier with standard params."""
    return CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6, verbose=0, allow_writing_files=False, random_state=42
    )
