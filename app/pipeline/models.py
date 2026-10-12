"""Model builders for RUL regression."""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor


def build_model(name: str, params: dict | None = None):
    params = params or {}

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 12),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            max_features=params.get("max_features", 0.8),
            random_state=42,
            n_jobs=1,
        )

    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 150),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )

    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 150),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")


MODEL_CANDIDATES = {
    "Random Forest": [{"n_estimators": 120, "max_depth": 12, "max_features": 0.8}],
    "XGBoost": [{"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05}],
    "Gradient Boosting": [{"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05}],
}
