"""Generate synthetic customers to exercise predict.py.

Important scope note: this data is for PLUMBING only, checking that predict.py
turns raw input into sensible probabilities without breaking. It is NOT for
validating the model. The labels are not generated at all (there is no Churn
column), because a model cannot be honestly validated on data whose answers
were invented. The held-out test split in model_training is the only source of
performance numbers. See the project README for the reasoning.

The output is RAW, matching the shape of the original Telco file (text
TotalCharges, "No internet service" spelled out, unencoded categories, a
customerID column), so predict.py's shared cleaning and encoding transforms
get genuinely exercised. A few deliberate edge-case rows are seeded in to poke
the corners (tenure-0 new customer, all services, no services).

Reproducible: a fixed seed means the committed example batch is stable.
"""

import os

import numpy as np
import pandas as pd

OUTPUT_PATH = os.path.join("data", "test", "new_customers.csv")
RANDOM_STATE = 42
N_CUSTOMERS = 200

SERVICE_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"[save] removed existing {path}")


def random_customers(n, rng):
    """Draw plausible customers from roughly realistic distributions. The
    internet add-ons are made conditional on having internet, so the generated
    data carries the same 'No internet service' structure as the real file and
    predict.py's collapse step gets exercised."""
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])

    rows = {
        "customerID": [f"GEN-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(1, 73, n),
        "PhoneService": phone,
        "MultipleLines": [
            "No phone service" if p == "No" else rng.choice(["Yes", "No"])
            for p in phone
        ],
        "InternetService": internet,
        "Contract": rng.choice(
            ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(18.0, 120.0, n).round(2),
    }

    # Internet add-ons: "No internet service" when there is no internet,
    # otherwise a real Yes/No choice.
    for col in SERVICE_ADDONS:
        rows[col] = [
            "No internet service" if net == "No" else rng.choice(["Yes", "No"])
            for net in internet
        ]

    df = pd.DataFrame(rows)

    # TotalCharges as text (matching the raw file), roughly tenure x monthly.
    total = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df["TotalCharges"] = total.astype(str)
    return df


def edge_cases():
    """Deliberate corner cases to stress the transforms."""
    base = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    }
    all_serv = {c: "Yes" for c in SERVICE_ADDONS}
    no_serv = {c: "No internet service" for c in SERVICE_ADDONS}

    rows = [
        # Brand-new customer: tenure 0, blank TotalCharges (the raw-file trap).
        {**base, "customerID": "EDGE-tenure0", "tenure": 0,
         "InternetService": "Fiber optic", "Contract": "Month-to-month",
         "MonthlyCharges": 70.0, "TotalCharges": " ", **{c: "No" for c in SERVICE_ADDONS}},
        # Every service, long tenure, two-year contract (should score low risk).
        {**base, "customerID": "EDGE-allservices", "tenure": 72,
         "InternetService": "Fiber optic", "Contract": "Two year",
         "MonthlyCharges": 118.0, "TotalCharges": "8496.0", **all_serv},
        # No internet at all (all add-ons become "No internet service").
        {**base, "customerID": "EDGE-noservices", "tenure": 12,
         "InternetService": "No", "MultipleLines": "No",
         "Contract": "One year", "MonthlyCharges": 20.0,
         "TotalCharges": "240.0", **no_serv},
    ]
    return pd.DataFrame(rows)


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_if_exists(path)
    df.to_csv(path, index=False)
    print(f"[save] {path}: {df.shape[0]} customers ({df.shape[1]} columns)")


def run(output_path=OUTPUT_PATH, n=N_CUSTOMERS):
    rng = np.random.default_rng(RANDOM_STATE)
    customers = random_customers(n, rng)
    edges = edge_cases()
    df = pd.concat([customers, edges], ignore_index=True)
    # Column order matching the raw file's spirit (id first, no Churn column).
    print(f"[generate] {len(customers)} random + {len(edges)} edge-case customers")
    save(df, output_path)
    return df


if __name__ == "__main__":
    run()
