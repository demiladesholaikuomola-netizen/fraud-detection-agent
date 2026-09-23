"""
ACTION AGENT — standalone version
Agent-Based Financial Fraud Detection and Prevention System

Role in the pipeline:
    The Action Agent receives a transaction after the Decision Agent has
    produced final_decision. It carries out the appropriate response:

        block   -> stop the transaction and raise an alert
        flag    -> allow the transaction to continue but mark it for review
        approve -> allow the transaction to proceed normally

Input expected:
    A transaction dictionary containing at least:
        transaction_id      (optional, but recommended)
        final_decision      ('block', 'flag', or 'approve')

Output produced:
    The same transaction dictionary with these fields added/updated:
        action_taken        ('blocked', 'flagged_for_review', or 'approved')
        transaction_status  ('blocked', 'pending_review', or 'approved')
        alert_raised        (True/False)
        action_message      (human-readable explanation)

The agent also provides a simple in-memory audit log so the team can
inspect what action was taken for each transaction during testing.
"""

from datetime import datetime


VALID_DECISIONS = {"block", "flag", "approve"}


# Keeps a record of actions during the current program run.
# In a real banking system this would normally be written to a database.
ACTION_LOG = []


def take_action(transaction):
    """
    Execute the response selected by the Decision Agent.

    Parameters
    ----------
    transaction : dict
        Transaction dictionary containing final_decision.

    Returns
    -------
    dict
        The same transaction dictionary with Action Agent fields added.
    """

    if not isinstance(transaction, dict):
        raise TypeError("Action agent expects a transaction dictionary.")

    if "final_decision" not in transaction:
        raise ValueError(
            "Action agent requires 'final_decision' to already be present "
            "before this agent runs."
        )

    decision = transaction["final_decision"]

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Unexpected final_decision value: '{decision}'. "
            "Must be exactly one of 'block', 'flag', or 'approve'."
        )

    transaction_id = transaction.get("transaction_id", "UNKNOWN")

    if decision == "block":
        action_taken = "blocked"
        transaction_status = "blocked"
        alert_raised = True
        action_message = (
            "Transaction blocked because the Decision Agent classified it as fraud. "
            "A fraud alert has been raised."
        )

    elif decision == "flag":
        action_taken = "flagged_for_review"
        transaction_status = "pending_review"
        alert_raised = True
        action_message = (
            "Transaction flagged for manual review because it was classified as suspicious. "
            "The transaction is not automatically blocked."
        )

    else:  # decision == "approve"
        action_taken = "approved"
        transaction_status = "approved"
        alert_raised = False
        action_message = "Transaction approved and allowed to proceed normally."

    transaction["action_taken"] = action_taken
    transaction["transaction_status"] = transaction_status
    transaction["alert_raised"] = alert_raised
    transaction["action_message"] = action_message
    transaction["action_timestamp"] = datetime.now().isoformat(timespec="seconds")

    # Store a compact audit record rather than copying the entire transaction.
    ACTION_LOG.append(
        {
            "transaction_id": transaction_id,
            "final_decision": decision,
            "action_taken": action_taken,
            "transaction_status": transaction_status,
            "alert_raised": alert_raised,
            "action_timestamp": transaction["action_timestamp"],
        }
    )

    return transaction


def get_action_log():
    """Return a copy of the current Action Agent audit log."""
    return ACTION_LOG.copy()


def clear_action_log():
    """Clear the in-memory audit log. Useful between test runs."""
    ACTION_LOG.clear()


if __name__ == "__main__":
    import json

    # Demo transactions matching the output of the Decision Agent.
    test_cases = [
        {
            "transaction_id": "TXN1001",
            "ml_score": 0.95,
            "rules_verdict": "block",
            "final_decision": "block",
        },
        {
            "transaction_id": "TXN1002",
            "ml_score": 0.84,
            "rules_verdict": "flag",
            "final_decision": "flag",
        },
        {
            "transaction_id": "TXN1003",
            "ml_score": 0.20,
            "rules_verdict": "clear",
            "final_decision": "approve",
        },
    ]

    print("=== ACTION AGENT DEMO ===")

    for test_transaction in test_cases:
        result = take_action(test_transaction.copy())
        print(json.dumps(result, indent=2))

    print("\n=== ACTION AUDIT LOG ===")
    print(json.dumps(get_action_log(), indent=2))
