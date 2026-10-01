
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_churn_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_churn(features: list[float]) -> dict:
    expected_features = list(model.feature_names_in_)

    if len(features) != len(expected_features):
        raise ValueError(
            f"Expected {len(expected_features)} features, "
            f"but received {len(features)}."
        )

    input_data = pd.DataFrame(
        [features],
        columns=expected_features
    )

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4)
    }
