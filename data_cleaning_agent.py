"""
Data Cleaning Agent
--------------------
First stage of the fraud detection pipeline. Validates and repairs a raw
transaction dictionary before it is passed on to the pattern spotting and
rule checking agents.

This agent ensures the seven shared format fields are present and
correctly formatted, and adds one new field, "is_clean". It also keeps
five raw PaySim fields alongside the shared format fields, since the
pattern spotting agent's trained model needs them to produce a real
score. These five fields are passed through unchanged, not validated
or reformatted, they are only carried forward so the next agent has
what it needs.
"""

from datetime import datetime, timedelta

# The raw PaySim CSV has exactly these columns: step, type, amount,
# nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest,
# newbalanceDest, isFraud, isFlaggedFraud. Three fields the team's
# format requires (transaction_id, timestamp, receiver_type) are not
# present in PaySim and must be derived, not just renamed.
PAYSIM_SIMULATION_START = datetime(2026, 1, 1)  # arbitrary reference point

# These five raw PaySim fields are not part of the shared format, but
# the pattern spotting agent's model was trained on them and cannot
# produce a meaningful score without them. They are carried forward
# untouched, in addition to the seven shared format fields.
PASSTHROUGH_FIELDS = [
    "type",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def map_paysim_row(row, row_index):
    """
    Converts one raw PaySim row into the shared transaction format.

    transaction_id is built from the row's position, since PaySim has
    no ID column. timestamp is derived from "step" (1 step = 1 hour)
    against an arbitrary start date, since PaySim has no real dates.
    receiver_type is worked out from nameDest, since any account
    starting with "M" is a merchant, which PaySim treats as external.

    The five fields listed in PASSTHROUGH_FIELDS are also copied
    across unchanged, so the pattern spotting agent has what it needs
    further down the pipeline.
    """
    receiver_id = row.get("nameDest")
    receiver_type = "external" if isinstance(receiver_id, str) and receiver_id.startswith("M") else "internal"
    timestamp = PAYSIM_SIMULATION_START + timedelta(hours=row.get("step", 0))

    transaction = {
        "transaction_id": f"TXN{row_index:08d}",
        "sender_id": row.get("nameOrig"),
        "receiver_id": receiver_id,
        "receiver_type": receiver_type,
        "amount": row.get("amount"),
        "sender_balance_before": row.get("oldbalanceOrg"),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for field in PASSTHROUGH_FIELDS:
        transaction[field] = row.get(field)

    return transaction


# The seven fields every transaction must contain, exactly as named
# in the team's shared transaction format.
REQUIRED_FIELDS = [
    "transaction_id",
    "sender_id",
    "receiver_id",
    "receiver_type",
    "amount",
    "sender_balance_before",
    "timestamp",
]

VALID_RECEIVER_TYPES = {"internal", "external"}

# The exact timestamp format produced by map_paysim_row(). PaySim itself
# has no date field, so this is the only format clean_transaction() ever
# needs to check.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Tracks transaction_ids already seen in this process, so repeated
# records can be caught. Cleared by reset_seen_transactions().
_seen_transaction_ids = set()


def reset_seen_transactions():
    """
    Clears the duplicate-tracking memory used by clean_transaction().

    Call this at the start of a fresh batch or test run so that
    transaction_ids from a previous run are not mistaken for duplicates.
    """
    _seen_transaction_ids.clear()


def _normalise_amount(value):
    """Converts a numeric-looking amount into a float, or None if invalid."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _normalise_timestamp(value):
    """Converts a valid timestamp string into itself, or None if malformed."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value.strip(), TIMESTAMP_FORMAT)
        return parsed.strftime(TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _normalise_receiver_type(value):
    """Converts receiver_type into lowercase 'internal' or 'external', or None."""
    if not isinstance(value, str):
        return None
    normalised = value.strip().lower()
    return normalised if normalised in VALID_RECEIVER_TYPES else None


def clean_transaction(transaction):
    """
    Validates and repairs one raw transaction dictionary.

    Fixes common formatting issues such as stray whitespace,
    string-formatted numbers, inconsistent date formats, and mixed
    casing. Flags records that are duplicated or missing required
    data as unrecoverable rather than guessing values for them.
    """
    transaction = dict(transaction)  # avoid mutating the caller's copy
    is_clean = True

    # 1. transaction_id must exist and be unique within this run.
    txn_id = transaction.get("transaction_id")
    if not txn_id or not isinstance(txn_id, str):
        is_clean = False
    elif txn_id in _seen_transaction_ids:
        is_clean = False
    else:
        _seen_transaction_ids.add(txn_id)

    # 2. sender_id / receiver_id must be present; normalise casing.
    for id_field in ("sender_id", "receiver_id"):
        value = transaction.get(id_field)
        if not value or not isinstance(value, str):
            is_clean = False
        else:
            transaction[id_field] = value.strip().upper()

    # 3. receiver_type must be exactly "internal" or "external".
    receiver_type = _normalise_receiver_type(transaction.get("receiver_type"))
    if receiver_type is None:
        is_clean = False
    else:
        transaction["receiver_type"] = receiver_type

    # 4. amount must be a positive number.
    amount = _normalise_amount(transaction.get("amount"))
    if amount is None or amount <= 0:
        is_clean = False
    else:
        transaction["amount"] = amount

    # 5. sender_balance_before must be a non-negative number.
    balance = _normalise_amount(transaction.get("sender_balance_before"))
    if balance is None or balance < 0:
        is_clean = False
    else:
        transaction["sender_balance_before"] = balance

    # 6. timestamp must be present and parseable.
    timestamp = _normalise_timestamp(transaction.get("timestamp"))
    if timestamp is None:
        is_clean = False
    else:
        transaction["timestamp"] = timestamp

    # 7. Confirm every required field survived cleaning.
    for field in REQUIRED_FIELDS:
        if field not in transaction:
            is_clean = False

    transaction["is_clean"] = is_clean
    return transaction


if __name__ == "__main__":
    # Quick manual check using a real PaySim-shaped row, run through the
    # actual mapping step so this reflects the real data path.
    paysim_row = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 1060.31,
        "nameOrig": "C429214117",
        "oldbalanceOrg": 1089.0,
        "newbalanceOrig": 28.69,
        "nameDest": "M1591654462",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }
    transaction = map_paysim_row(paysim_row, row_index=0)
    result = clean_transaction(transaction)
    print(result)

    # Confirm the fields the pattern spotting agent needs are present
    for field in PASSTHROUGH_FIELDS:
        assert field in result, f"missing field the pattern agent needs: {field}"
    print("all pattern spotting fields present:", {f: result[f] for f in PASSTHROUGH_FIELDS})
