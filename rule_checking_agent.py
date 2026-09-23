# Threshold for a very large transaction.
#
# Set from the actual PaySim distribution rather than a guess: on a
# 200k-row sample, normal transactions had a median of ~N68,721 and a
# 99th percentile of ~N1,769,210, while confirmed fraud had a LOWER
# median amount (~N43,092) than normal transactions. This means "large
# amount" is a weak, rare signal here, not a strong fraud indicator --
# fraud in PaySim is identified by HOW money moves (full balance
# drains, mismatched destination balances) not by size. Setting this
# near the 99th percentile of normal transactions means it only fires
# on genuinely unusual amounts instead of flagging ~40% of everything.
LARGE_AMOUNT_THRESHOLD = 1_770_000


def check_blacklisted_account(transaction:dict, blacklist)->bool:
    """
    Checks whether the sender or receiver is blacklisted.
    """
    sender_id = transaction["sender_id"]
    receiver_id = transaction["receiver_id"]

    return sender_id in blacklist or receiver_id in blacklist


def check_large_amount(transaction)->bool:
    """
    Checks whether the transaction amount is
    N100,000 or more.
    """
    amount = int(transaction["amount"])

    return amount >= LARGE_AMOUNT_THRESHOLD


def check_account_fully_drained(transaction)->bool:
    """
    Checks whether the transaction uses the sender's
    entire available balance.
    """
    amount = transaction["amount"]
    balance = transaction["sender_balance_before"]

    return balance > 0 and amount == balance

def check_odd_hour_transaction(transaction):
    """
    Checks whether a transaction happened between
    12:00 AM and 5:00 AM.
    """
    time = int(transaction["timestamp"].split()[1].split(":")[0])

    return time == 0 or time <= 4


# Number of transactions from the same sender within the velocity
# window (see orchestrator.py) that counts as suspiciously fast,
# repeated activity -- a real compliance signal (Section 3 of the
# proposal names "velocity limits" explicitly as a Compliance Agent
# check). 3+ transactions in a short window is a common real-world
# starting point for mobile money velocity rules.
VELOCITY_COUNT_THRESHOLD = 3


def check_high_velocity(transaction) -> bool:
    """
    Checks whether the sender has made several transactions in quick
    succession. This agent holds no state itself -- it relies on
    'recent_txn_count' being pre-computed and attached to the
    transaction by the orchestrator, which tracks each sender's
    recent activity as transactions stream through the pipeline
    (the same pattern already used for the blacklist parameter).
    Defaults to 0 (never triggers) if the orchestrator didn't supply
    it, so this stays safe to call standalone or in tests.
    """
    return transaction.get("recent_txn_count", 0) >= VELOCITY_COUNT_THRESHOLD


def rule_checking_agent(transaction, blacklist=None)->dict:
    """
    Applies fixed fraud-detection rules to a cleaned transaction.

    The function keeps all existing transaction fields unchanged
    and adds only:

        transaction["rules_triggered"]
        transaction["rules_verdict"]

    Possible verdicts:
        "block"
        "flag"
        "clear"
    """

    # If no blacklist is supplied, use an empty set
    if blacklist is None:
        blacklist = set()

    blacklist = set(blacklist)

    rules_triggered = []

    # Rule 1: Blacklisted account
    # Verdict: BLOCK
    if check_blacklisted_account(transaction, blacklist):
        rules_triggered.append("blacklisted_account")

    # Rule 2: Large amount
    # Verdict: FLAG
    if check_large_amount(transaction):
        rules_triggered.append("large_amount")
    
    # Rule 3: Account fully drained
    # Verdict: FLAG
    if check_account_fully_drained(transaction):
        rules_triggered.append("account_fully_drained")

    # Rule 4: Transaction at odd hours
    # Verdict: BLOCK only when paired with large_amount (see below)
    if check_odd_hour_transaction(transaction):
        rules_triggered.append("odd_hour_transaction")

    # Rule 5: Sender making several transactions in quick succession
    # Verdict: FLAG
    if check_high_velocity(transaction):
        rules_triggered.append("high_velocity")

    # Determine rules verdict
    # - Blacklisted account is always an automatic block.
    # - Odd-hour alone is common for legitimate night-time mobile money
    #   use, so on its own it only flags. It escalates to a block when
    #   it co-occurs with a large amount, which is a much stronger
    #   combined signal (large sums moving at 12am-4:59am).
    # - High velocity alone only flags (could be a legitimate business
    #   account); it escalates to a block when paired with a large
    #   amount, since fast + big is a stronger mule-account signal
    #   than either alone.
    if "blacklisted_account" in rules_triggered:
        rules_verdict = "block"
    elif "odd_hour_transaction" in rules_triggered and "large_amount" in rules_triggered:
        rules_verdict = "block"
    elif "high_velocity" in rules_triggered and "large_amount" in rules_triggered:
        rules_verdict = "block"
    elif rules_triggered:
        rules_verdict = "flag"
    else:
        rules_verdict = "clear"

    transaction["rules_triggered"] = rules_triggered
    transaction["rules_verdict"] = rules_verdict

    return transaction

if __name__ == "__main__":
    from pprint import pprint

    transaction = {
    "transaction_id": "TXN10293",
    "sender_id": "USR001",
    "receiver_id": "USR045",
    "receiver_type": "internal",   # "internal" or "external"
    "amount": 250000,
    "sender_balance_before": 250000,
    "timestamp": "2026-08-16 0:00:00"
    }


    pprint(rule_checking_agent(transaction))

    #Example ouput
    {'amount': 250000,
    'receiver_id': 'USR045',
    'receiver_type': 'internal',
    'rules_triggered': ['large_amount', 'account_fully_drained', 'odd_hour_transaction'],
    'rules_verdict': 'block',
    'sender_balance_before': 250000,
    'sender_id': 'USR001',
    'timestamp': '2026-08-16 0:00:00',
    'transaction_id': 'TXN10293'}