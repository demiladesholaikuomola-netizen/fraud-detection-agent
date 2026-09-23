"""
Pattern Spotting Agent (Agent #2)

Job: receive a cleaned transaction dictionary (using PaySim's own
column names, as agreed by the team), add an "ml_score" field
(a number from 0.0 to 1.0) representing how unusual/suspicious the
transaction looks, and return the dictionary unchanged otherwise.

Expected input dictionary (from the Data Cleaning Agent), using
PaySim's exact column names:
{
    "step": 1,
    "type": "TRANSFER",              # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    "amount": 181.0,
    "nameOrig": "C1305486145",
    "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C553264065",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "isFraud": 0,          # may or may not be present depending on stage
    "isFlaggedFraud": 0,   # may or may not be present depending on stage
    "is_clean": True       # added by Data Cleaning Agent
}

Model: a Random Forest classifier trained on the PaySim mobile-money
fraud dataset (6.3 million transactions, 8,213 of them real fraud).
"""

import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pattern_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_cols.pkl")

_model = joblib.load(MODEL_PATH)
_feature_cols = joblib.load(FEATURES_PATH)

_ALL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def _build_features(transaction: dict) -> pd.DataFrame:
    """
    Turns one PaySim-style transaction dictionary into the row of
    numbers the model expects.
    """
    amount = transaction.get("amount", 0)
    old_orig = transaction.get("oldbalanceOrg", 0)
    new_orig = transaction.get("newbalanceOrig", 0)
    old_dest = transaction.get("oldbalanceDest", 0)
    new_dest = transaction.get("newbalanceDest", 0)
    txn_type = transaction.get("type", "TRANSFER")

    orig_emptied = int(new_orig == 0 and old_orig > 0)
    dest_balance_unchanged = int(old_dest == 0 and new_dest == 0 and amount > 0)
    balance_error_orig = old_orig - amount - new_orig

    row = {
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "orig_emptied": orig_emptied,
        "dest_balance_unchanged": dest_balance_unchanged,
        "balance_error_orig": balance_error_orig,
    }

    for t in _ALL_TYPES:
        row[f"type_{t}"] = int(txn_type == t)

    return pd.DataFrame([row])[_feature_cols]


def _build_features_batch(transactions: list) -> pd.DataFrame:
    """
    Same feature construction as _build_features, but for a whole
    list of transactions at once, so the model can be called once on
    a big DataFrame instead of once per row.
    """
    rows = []
    for transaction in transactions:
        amount = transaction.get("amount", 0)
        old_orig = transaction.get("oldbalanceOrg", 0)
        new_orig = transaction.get("newbalanceOrig", 0)
        old_dest = transaction.get("oldbalanceDest", 0)
        new_dest = transaction.get("newbalanceDest", 0)
        txn_type = transaction.get("type", "TRANSFER")

        orig_emptied = int(new_orig == 0 and old_orig > 0)
        dest_balance_unchanged = int(old_dest == 0 and new_dest == 0 and amount > 0)
        balance_error_orig = old_orig - amount - new_orig

        row = {
            "amount": amount,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "orig_emptied": orig_emptied,
            "dest_balance_unchanged": dest_balance_unchanged,
            "balance_error_orig": balance_error_orig,
        }
        for t in _ALL_TYPES:
            row[f"type_{t}"] = int(txn_type == t)
        rows.append(row)

    return pd.DataFrame(rows)[_feature_cols]


def spot_pattern_batch(transactions: list) -> list:
    """
    Same job as spot_pattern(), but for a whole list of transaction
    dicts at once: builds one big feature table and calls the model
    a single time, instead of once per transaction.

    Calling a scikit-learn model repeatedly on single rows carries
    fixed per-call overhead (thread-pool setup across the forest's
    estimators, input validation, etc.) that adds up fast at PaySim's
    scale -- this is what made the row-by-row version slow on a full
    dataset. Batching removes that per-row tax; the maths per row is
    identical, so ml_score values match spot_pattern() exactly.

    Mutates and returns the same list of dicts, each now carrying
    ml_score, same as spot_pattern() does for one transaction.
    """
    if not transactions:
        return transactions

    features = _build_features_batch(transactions)
    fraud_probabilities = _model.predict_proba(features)[:, 1]

    for transaction, probability in zip(transactions, fraud_probabilities):
        transaction["ml_score"] = round(float(probability), 4)

    return transactions


def spot_pattern(transaction: dict) -> dict:
    """
    Receives: the cleaned transaction dictionary (PaySim field names).

    Returns: the same dictionary, plus transaction["ml_score"],
    a float from 0.0 to 1.0 -- 0 means the transaction looks
    completely normal, 1 means it looks highly likely to be fraud.
    """
    features = _build_features(transaction)
    fraud_probability = _model.predict_proba(features)[0][1]

    transaction["ml_score"] = round(float(fraud_probability), 4)
    return transaction


if __name__ == "__main__":
    # Normal-looking transaction
    normal_transaction = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "is_clean": True,
    }
    result1 = spot_pattern(normal_transaction)
    print("Normal transaction ml_score:", result1["ml_score"])

    # Real fraud example, taken directly from the dataset
    fraud_transaction = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "is_clean": True,
    }
    result2 = spot_pattern(fraud_transaction)
    print("Known fraud transaction ml_score:", result2["ml_score"])
