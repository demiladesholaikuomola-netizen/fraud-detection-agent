# learning_agent.py
#
# LEARNING AGENT — standalone version
# Agent-Based Financial Fraud Detection and Prevention System
#
# Role in the pipeline:
#   The learning agent doesn't sit in the per-transaction pipeline
#   like the other five. It runs periodically on a BATCH of
#   already-decided transactions (output of the Decision Agent),
#   compares each final_decision against the transaction's true label
#   (fraud / not fraud, taken from the Kaggle dataset), and works out
#   how the system's threshold should shift to improve.
#
#   For this project, "improve the system" means: recommend a new
#   ML_SCORE_THRESHOLD value that the Decision Agent should switch to
#   next time, based on how many false alarms vs missed fraud cases
#   showed up in the batch. A live, continuously retraining model is
#   out of scope — demonstrating one adjustment pass on held-out data
#   is enough to prove the concept (per the project guide, section 2.6).
#
# Input this agent expects:
#   A batch (list) of transaction dicts, each already carrying the
#   fields the Decision Agent produces:
#       ml_score, rules_verdict, final_decision  ("block"/"flag"/"approve")
#   plus one ground-truth field this agent relies on:
#       actual_fraud   (True/False — the real Kaggle label)
#
# Output this agent produces:
#   A report dict: confusion counts, false positive/negative rates,
#   and a recommended new ML_SCORE_THRESHOLD.
#
# This file does NOT import decision_agent.py or anything else from
# the team. It only needs whatever transaction batch the Decision
# Agent already produced — hand it that, and it runs on its own.


CURRENT_ML_SCORE_THRESHOLD = 0.8  # whatever the Decision Agent is using right now


# ---------------------------------------------------------------------
# 1. EVALUATE A BATCH OF DECIDED TRANSACTIONS
# ---------------------------------------------------------------------
def evaluate_batch(decided_transactions):
    """
    Compares final_decision against actual_fraud for every transaction
    in the batch and returns confusion-matrix-style counts, PLUS
    naira-value-weighted totals.

    Counts alone treat a missed N500 fraud the same as a missed
    N5,000,000 fraud, which isn't how a real fraud desk would judge
    the system. Alongside true_positive/false_positive/etc counts,
    this also reports:
        fraud_value_caught     -- total amount of fraud correctly stopped
        fraud_value_missed     -- total amount of fraud that got through
        legitimate_value_disrupted -- total amount wrongly flagged/blocked

    A transaction counts as "flagged" if final_decision is
    'block' or 'flag' (either one interrupts a normal transaction).
    'approve' counts as "cleared". amount defaults to 0 if a
    transaction dict doesn't carry one, so this stays backward
    compatible with batches that predate this field.

    Beyond that combined view, the report also splits false/true
    positives by final_decision, since 'block' and 'flag' carry very
    different real-world costs: a block stops a legitimate customer's
    transaction outright, while a flag only queues it for a human
    analyst to review (see storage.py's review_queue). Lumping them
    together overstates how many customers are actually blocked.
    """
    true_positive = 0   # flagged, and it really was fraud
    false_positive = 0  # flagged, but it was actually normal
    true_negative = 0   # cleared, and it really was normal
    false_negative = 0  # cleared, but it was actually fraud

    fraud_value_caught = 0.0
    fraud_value_missed = 0.0
    legitimate_value_disrupted = 0.0

    # Breakdown by exact final_decision, for the block-vs-flag cost story.
    blocked_true_positive = 0    # correctly blocked real fraud
    blocked_false_positive = 0   # wrongly blocked a legitimate transaction
    flagged_true_positive = 0    # real fraud sent to review, not blocked
    flagged_false_positive = 0   # legitimate transaction sent to review

    for txn in decided_transactions:
        decision = txn["final_decision"]
        flagged = decision in ("block", "flag")
        actually_fraud = txn["actual_fraud"]
        amount = txn.get("amount", 0) or 0

        if flagged and actually_fraud:
            true_positive += 1
            fraud_value_caught += amount
            if decision == "block":
                blocked_true_positive += 1
            else:
                flagged_true_positive += 1
        elif flagged and not actually_fraud:
            false_positive += 1
            legitimate_value_disrupted += amount
            if decision == "block":
                blocked_false_positive += 1
            else:
                flagged_false_positive += 1
        elif not flagged and not actually_fraud:
            true_negative += 1
        else:
            false_negative += 1
            fraud_value_missed += amount

    total = len(decided_transactions)
    return {
        "total": total,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "false_positive_rate": round(false_positive / total, 3) if total else 0,
        "false_negative_rate": round(false_negative / total, 3) if total else 0,
        "fraud_value_caught": round(fraud_value_caught, 2),
        "fraud_value_missed": round(fraud_value_missed, 2),
        "legitimate_value_disrupted": round(legitimate_value_disrupted, 2),
        "blocked_true_positive": blocked_true_positive,
        "blocked_false_positive": blocked_false_positive,
        "flagged_true_positive": flagged_true_positive,
        "flagged_false_positive": flagged_false_positive,
    }


# ---------------------------------------------------------------------
# 2. ADJUST THE THRESHOLD BASED ON THE EVALUATION
# ---------------------------------------------------------------------
# Ceiling for the recommended threshold. Deliberately kept below 1.0:
# ml_score can never exceed 1.0, so a threshold of exactly 1.0 makes
# the condition "ml_score > threshold" permanently False, silently
# turning the whole ML model into a no-op for every future decision
# -- the Decision Agent would then be arbitrating using rules alone,
# which defeats the proposal's core argument for combining both
# signals (Section 2). Capping below 1.0 keeps the ML model able to
# matter, however rarely, even after the threshold has climbed a lot.
MAX_ML_SCORE_THRESHOLD = 0.97

# Floor for the same reason in the other direction -- stops the
# system from ever becoming so permissive that almost every score
# clears it.
MIN_ML_SCORE_THRESHOLD = 0.3

# Neutral reference point. When a batch's false positive AND false
# negative rates are both already acceptable, the threshold gently
# drifts back toward this value instead of staying frozen wherever an
# earlier, noisier batch happened to push it. Without this, a
# threshold that once spiked (e.g. from one unusually fraud-heavy
# batch) would stay at that extreme value forever, since the fpr/fnr
# comparisons alone give it no way back once things stabilise.
BASELINE_ML_SCORE_THRESHOLD = 0.8


def recommend_threshold(current_threshold, metrics, step=0.05, drift_step=0.01):
    """
    Simple, explainable adjustment rule:
      - too many false alarms (false_positive_rate too high)
            -> raise the threshold, make it harder to flag/block
      - too much fraud slipping through (false_negative_rate too high)
            -> lower the threshold, make it easier to flag/block
      - both are low -> drift gently back toward a neutral baseline

    The 0.1 tolerance below is a placeholder; in Chapter 4 this should
    be justified against the Kaggle dataset's actual score spread.
    """
    fpr = metrics["false_positive_rate"]
    fnr = metrics["false_negative_rate"]
    new_threshold = current_threshold

    if fpr > 0.1 and fpr >= fnr:
        new_threshold = round(min(current_threshold + step, MAX_ML_SCORE_THRESHOLD), 3)
        reason = f"false positive rate {fpr} is too high raising threshold to catch fewer normal transactions"
    elif fnr > 0.1 and fnr > fpr:
        new_threshold = round(max(current_threshold - step, MIN_ML_SCORE_THRESHOLD), 3)
        reason = f"false negative rate {fnr} is too high lowering threshold to catch more fraud"
    else:
        if current_threshold > BASELINE_ML_SCORE_THRESHOLD:
            new_threshold = round(max(current_threshold - drift_step, BASELINE_ML_SCORE_THRESHOLD), 3)
        elif current_threshold < BASELINE_ML_SCORE_THRESHOLD:
            new_threshold = round(min(current_threshold + drift_step, BASELINE_ML_SCORE_THRESHOLD), 3)
        reason = (
            f"false positive rate {fpr} and false negative rate {fnr} are both "
            f"acceptable; drifting threshold toward baseline {BASELINE_ML_SCORE_THRESHOLD}"
        )

    return new_threshold, reason


# ---------------------------------------------------------------------
# 3. RUN ONE FULL LEARNING PASS
# ---------------------------------------------------------------------
def learning_agent(batch, current_threshold):
    """
    Full cycle: evaluate the batch under the current threshold, then
    recommend an adjusted threshold for next time.
    """
    metrics = evaluate_batch(batch)
    new_threshold, reason = recommend_threshold(current_threshold, metrics)

    return {
        "evaluated_on": metrics["total"],
        "metrics": metrics,
        "old_threshold": current_threshold,
        "new_threshold": new_threshold,
        "reason": reason,
    }


# ---------------------------------------------------------------------
# 4. DEMO RUN — a held-out batch, standing in for what the
#    Decision Agent would have already handed off
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # Each transaction here already has final_decision filled in —
    # exactly what this agent would receive from the Decision Agent
    # in the real pipeline. actual_fraud is the ground-truth label
    # that would come from the Kaggle dataset.
    decided_batch = [
        {"ml_score": 0.92, "rules_verdict": "flag",  "final_decision": "block",   "actual_fraud": True},
        {"ml_score": 0.55, "rules_verdict": "clear", "final_decision": "approve", "actual_fraud": False},
        {"ml_score": 0.88, "rules_verdict": "clear", "final_decision": "flag",    "actual_fraud": False},  # false alarm
        {"ml_score": 0.83, "rules_verdict": "clear", "final_decision": "flag",    "actual_fraud": False},  # false alarm
        {"ml_score": 0.15, "rules_verdict": "clear", "final_decision": "approve", "actual_fraud": False},
        {"ml_score": 0.20, "rules_verdict": "block", "final_decision": "block",   "actual_fraud": True},
        {"ml_score": 0.10, "rules_verdict": "clear", "final_decision": "approve", "actual_fraud": True},   # missed fraud
        {"ml_score": 0.60, "rules_verdict": "clear", "final_decision": "approve", "actual_fraud": False},
        {"ml_score": 0.81, "rules_verdict": "clear", "final_decision": "flag",    "actual_fraud": False},  # false alarm
        {"ml_score": 0.05, "rules_verdict": "clear", "final_decision": "approve", "actual_fraud": False},
    ]

    report = learning_agent(decided_batch, CURRENT_ML_SCORE_THRESHOLD)
    print("=== LEARNING AGENT REPORT ===")
    print(json.dumps(report, indent=2))
