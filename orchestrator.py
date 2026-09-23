"""
orchestrator.py
================
The message bus / coordinator described in Section 3 of the proposal
("six agents coordinated by a lightweight orchestrator... following
the Actor-Model pattern"). This file does not re-implement any agent
logic -- it only calls the six agent modules in the right order and
passes each transaction's belief-dict from one agent to the next,
matching the Perceive -> Reason -> Act cycle in Section 3.1.

Pipeline order (per-transaction, real-time path):

    raw PaySim row
        -> data_cleaning_agent.map_paysim_row      (Perceive: build beliefs)
        -> data_cleaning_agent.clean_transaction   (Perceive: validate beliefs)
        -> [quarantine here if is_clean is False]
        -> pattern_agent.spot_pattern               (Reason: statistical)
        -> rule_checking_agent.rule_checking_agent  (Reason: symbolic/rules)
        -> decision_agent.make_decision             (Reason: arbitration)
        -> action_agent.take_action                 (Act)

The Learning Agent is NOT part of this per-transaction path (see its
own docstring: it runs on a batch). The orchestrator collects decided
transactions into a batch and periodically calls it, then feeds its
recommended threshold back into decision_agent for the next batch --
closing the feedback loop shown in Section 3.1's diagram.

Ground-truth handling
----------------------
PaySim's isFraud column is intentionally NOT carried through the
cleaning/pattern/rule/decision stages, so it can never leak into the
score or the rules (this is worth a line in the report under Section
9's "Data privacy & ethics" row). The orchestrator keeps it aside in
a separate dict, keyed by transaction_id, and only reattaches it as
actual_fraud when building the batch handed to the Learning Agent.
"""

from datetime import datetime
from collections import defaultdict, deque, Counter

import data_cleaning_agent as dca
import pattern_agent as pa
import rule_checking_agent as rca
import decision_agent as da
import action_agent as aa
import learning_agent as la
import storage as st


def build_blacklist_from_paysim(csv_path, min_fraud_count=2, chunksize=200_000, include_receiver=True):
    """
    Scans a PaySim CSV for accounts that show up repeatedly on
    confirmed-fraud (isFraud == 1) rows, and returns a set of account
    IDs to use as Orchestrator(blacklist=...).

    Checks both nameOrig (sender) and nameDest (receiver) by default,
    since PaySim fraud typically moves money through mule accounts
    that receive stolen funds before cashing out -- a single account
    is unlikely to appear twice as a *sender*, but may well appear as
    a *receiver* across several fraud cases.

    Note: with min_fraud_count=2, this may return a small or even
    empty set on PaySim, since much of its simulated fraud uses
    one-off compromised accounts. If that happens, it's worth
    reporting as a finding in Chapter 4 (repeat-offender blacklisting
    has limited value on this particular dataset) rather than a bug
    -- try min_fraud_count=1 to see the full one-time-offender list
    for comparison.
    """
    import pandas as pd
    from collections import Counter

    counts = Counter()
    cols = ["nameOrig", "isFraud"] + (["nameDest"] if include_receiver else [])

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=cols):
        fraud_rows = chunk[chunk["isFraud"] == 1]
        counts.update(fraud_rows["nameOrig"])
        if include_receiver:
            counts.update(fraud_rows["nameDest"])

    blacklist = {str(acct).strip().upper() for acct, c in counts.items() if c >= min_fraud_count}

    print(f"Scanned for repeat fraud accounts (min_fraud_count={min_fraud_count}): "
          f"{len(blacklist)} accounts blacklisted out of {len(counts)} seen in fraud rows.")

    return blacklist


def seed_demo_accounts(storage):
    """
    Sets up the fixed demo accounts for the virtual bank: four
    customer accounts to log into and send from (all but Tunde are
    Ivory Trust Bank), one of a different bank for the interbank
    demo, and one merchant account (M-prefixed, so the pipeline
    treats it as an external receiver, matching PaySim's own
    convention) as a payment destination. Safe to call every app
    startup -- seed_accounts only inserts accounts that don't already
    exist, so balances persist across restarts.

    Four accounts (Ada Obi, Chika Eze, Efe Okafor, Ngozi Adeyemi) sit
    on our own fictional bank, "Ivory Trust Bank" -- ordinary
    transfers between them, and enough logins for a small group to
    each hold their own account during a demo. Tunde Bello is
    deliberately on a different bank, "Horizon MFB", so sending to
    him in the customer app demonstrates a realistic interbank
    transfer (pick a different bank, enter an account number) rather
    than every transfer being within one bank.
    """
    storage.seed_accounts([
        {"account_id": "C100000001", "display_name": "Ada Obi", "pin": "1234",
         "bank_name": "Ivory Trust Bank", "available_balance": 500_000.0},
        {"account_id": "C100000002", "display_name": "Tunde Bello", "pin": "1234",
         "bank_name": "Horizon MFB", "available_balance": 250_000.0},
        {"account_id": "C100000003", "display_name": "Chika Eze", "pin": "1234",
         "bank_name": "Ivory Trust Bank", "available_balance": 3_000_000.0},
        {"account_id": "M100000004", "display_name": "Jumia Store (merchant)", "pin": None,
         "bank_name": "Ivory Trust Bank", "available_balance": 0.0},
        {"account_id": "C100000005", "display_name": "Efe Okafor", "pin": "1234",
         "bank_name": "Ivory Trust Bank", "available_balance": 750_000.0},
        {"account_id": "C100000006", "display_name": "Ngozi Adeyemi", "pin": "1234",
         "bank_name": "Ivory Trust Bank", "available_balance": 2_000_000.0},
    ])


def seed_demo_admins(storage):
    """
    Sets up the fixed analyst logins for the Fraud Ops Command
    Center. Separate from customer accounts entirely -- an analyst
    is bank staff, not a customer, and has no PIN, balance, or bank
    name of their own.
    """
    storage.seed_admin_users([
        {"username": "jadeyemi", "password": "fraudops2026", "display_name": "J. Adeyemi"},
        {"username": "cokonkwo", "password": "fraudops2026", "display_name": "C. Okonkwo"},
    ])


class Orchestrator:
    def __init__(self, blacklist=None, learn_every=500, db_path=None, velocity_window_steps=3):
        self.blacklist = set(blacklist) if blacklist else set()

        # Threshold currently in force. Starts at decision_agent's own
        # default and gets updated after each Learning Agent pass.
        self.current_threshold = da.ML_SCORE_THRESHOLD

        # How many decided transactions to collect before running a
        # Learning Agent pass. 500 is a placeholder -- tune against
        # your dataset size in Chapter 4.
        self.learn_every = learn_every

        # Velocity check window, in PaySim "step" units (1 step = 1
        # simulated hour). A sender with >= VELOCITY_COUNT_THRESHOLD
        # transactions within this many steps of each other trips the
        # high_velocity rule. Assumes the CSV is read in step order,
        # which the real PaySim file is.
        self.velocity_window_steps = velocity_window_steps
        # A single global sliding window of (step, sender_id) pairs,
        # plus a running count per sender within that window. This
        # replaces an earlier per-sender dict design that kept one
        # entry PER SENDER FOREVER, even after that sender's last
        # transaction aged out of the window -- across millions of
        # unique PaySim accounts that dict grew without bound for the
        # whole run, eventually exhausting memory and forcing Windows
        # to swap to disk (the 100% disk usage / slowdown after ~4.3M
        # rows). This version's memory use is bounded by how many
        # transactions occur within velocity_window_steps, not by how
        # many unique senders exist in the whole file.
        self._velocity_window = deque()
        self._velocity_counts = Counter()

        # Optional persistent log, per the proposal's SQLite storage
        # layer (Section 4). If no db_path is given, nothing is
        # written to disk -- useful for quick smoke tests.
        self.storage = st.PipelineStorage(db_path) if db_path else None

        # If this database already has a threshold learned from a
        # previous run (e.g. the historical evaluation pass), pick it
        # up instead of resetting to the hardcoded default. This is
        # what lets the live customer-app demo apply a threshold the
        # system actually learned from 5+ million real transactions,
        # rather than starting cold.
        if self.storage:
            persisted = self.storage.get_state("current_threshold")
            if persisted is not None:
                self.current_threshold = float(persisted)

        # Transactions that failed cleaning. Kept for the report's
        # "how much of the raw data was unusable" discussion.
        self.quarantined = []

        # Rolling batch of decided transactions (with actual_fraud
        # reattached), waiting for the next Learning Agent pass.
        self._pending_batch = []

        # Full history of every Learning Agent report produced, so
        # you can plot how the threshold moved over time.
        self.learning_history = []

    def process_rows(self, raw_rows, start_index):
        """
        Runs a whole list of raw PaySim rows through the pipeline,
        batching the Pattern Agent's ML scoring into a single model
        call instead of one call per row (see pattern_agent.py's
        spot_pattern_batch -- this is what makes a full 6.3M-row run
        practical instead of taking hours). Every other agent still
        runs per-transaction, since plain Python logic doesn't carry
        the same per-call overhead a scikit-learn model does.

        Returns the list of result dicts, in the same order as
        raw_rows (including quarantined ones).
        """
        cleaned = []
        results = [None] * len(raw_rows)

        # Pass 1: clean + quarantine. Anything dirty is finished here
        # and never reaches scoring/rules/decision/action.
        for i, raw_row in enumerate(raw_rows):
            transaction = dca.map_paysim_row(raw_row, start_index + i)
            transaction = dca.clean_transaction(transaction)

            actual_fraud = bool(raw_row.get("isFraud", 0))
            transaction.pop("isFraud", None)

            if not transaction["is_clean"]:
                self.quarantined.append(transaction)
                if self.storage:
                    self.storage.log(transaction, actual_fraud=actual_fraud)
                results[i] = transaction
                continue

            # Stash what later stages need but that isn't part of the
            # shared transaction format, so it travels with the dict
            # through batch scoring without touching ml_score/rules.
            transaction["_actual_fraud"] = actual_fraud
            transaction["_step"] = raw_row.get("step", 0)
            transaction["_result_index"] = i
            cleaned.append(transaction)

        # Pass 2: one batched ML scoring call for every clean
        # transaction in this chunk.
        cleaned = pa.spot_pattern_batch(cleaned)

        # Pass 3: velocity, rules, decision, action -- still
        # per-transaction, since these are cheap pure-Python checks.
        for transaction in cleaned:
            actual_fraud = transaction.pop("_actual_fraud")
            step = transaction.pop("_step")
            result_index = transaction.pop("_result_index")

            transaction["recent_txn_count"] = self._record_and_count_velocity(
                transaction["sender_id"], step
            )
            transaction = rca.rule_checking_agent(transaction, blacklist=self.blacklist)
            transaction = da.make_decision(transaction, ml_score_threshold=self.current_threshold)
            transaction = aa.take_action(transaction)

            if self.storage:
                self.storage.log(transaction, actual_fraud=actual_fraud)
                if transaction["final_decision"] == "flag":
                    self.storage.queue_for_review(transaction)

            self._pending_batch.append(
                {
                    "transaction_id": transaction["transaction_id"],
                    "amount": transaction["amount"],
                    "ml_score": transaction["ml_score"],
                    "rules_verdict": transaction["rules_verdict"],
                    "final_decision": transaction["final_decision"],
                    "actual_fraud": actual_fraud,
                }
            )
            if len(self._pending_batch) >= self.learn_every:
                self._run_learning_pass()

            results[result_index] = transaction

        return results

    def process_row(self, raw_row, row_index):
        """
        Runs one raw PaySim row through the full real-time pipeline.
        Returns the final transaction dict (after Action Agent), or
        the quarantined dict if it failed cleaning.

        Kept for single-transaction use (demos, tests, or a live
        one-at-a-time API) -- for processing a whole file, use
        process_rows()/run_on_csv() instead, which batch the ML
        scoring step and are dramatically faster at scale.
        """
        return self.process_rows([raw_row], row_index)[0]

    def process_live_transaction(self, sender_id, receiver_id, txn_type, amount):
        """
        Runs one customer-app-submitted transaction through the same
        six-agent pipeline as the historical evaluation path, but:

          - reads/writes real account balances (with held funds for
            'flag' verdicts) instead of scoring against a known label
          - logs to live_transactions, NOT the historical `transactions`
            table, so demo activity can never contaminate the Chapter
            4 evaluation numbers
          - does NOT feed into the Learning Agent's batch, since there
            is no ground truth for a made-up demo transaction

        Requires self.storage (accounts must exist there). Returns a
        dict with either an "error" key (e.g. insufficient funds) or
        the full transaction result the customer app can render.
        """
        if not self.storage:
            raise ValueError("process_live_transaction requires a db_path/storage to be set")

        sender = self.storage.get_account(sender_id)
        if sender is None:
            return {"error": "unknown_sender", "message": f"No account found: {sender_id}"}

        if amount <= 0:
            return {"error": "invalid_amount", "message": "Amount must be positive."}
        if amount > sender["available_balance"]:
            return {
                "error": "insufficient_funds",
                "message": f"Available balance is {sender['available_balance']:.2f}, "
                           f"which is less than {amount:.2f}.",
            }

        receiver = self.storage.get_account(receiver_id)
        sender_total_balance = sender["available_balance"] + sender["held_balance"]

        # Use the persisted simulated clock so timestamps keep moving
        # forward across separate app runs, the same way PaySim's own
        # 'step' field represents simulated hours.
        step = int(self.storage.get_state("current_step", 0)) + 1
        self.storage.set_state("current_step", step)

        raw_row = {
            "step": step,
            "type": txn_type,
            "amount": amount,
            "nameOrig": sender_id,
            "oldbalanceOrg": sender_total_balance,
            "newbalanceOrig": max(sender_total_balance - amount, 0),
            "nameDest": receiver_id,
            "oldbalanceDest": receiver["available_balance"] if receiver else 0,
            "newbalanceDest": (receiver["available_balance"] + amount) if receiver else amount,
        }

        transaction = dca.map_paysim_row(raw_row, step)
        transaction["transaction_id"] = f"LIVE{step:08d}"
        # Use the real current time for live transactions, not PaySim's
        # step-based simulated clock (which starts at midnight and
        # would make almost every early demo transaction spuriously
        # trip the odd-hour rule, since step 1-4 = 1am-4am).
        transaction["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transaction = dca.clean_transaction(transaction)
        transaction["type"] = txn_type  # map_paysim_row doesn't carry this into live IDs path

        if not transaction["is_clean"]:
            return {"error": "invalid_transaction", "message": "Transaction failed validation.", "detail": transaction}

        transaction = pa.spot_pattern_batch([transaction])[0]
        transaction["recent_txn_count"] = self._record_and_count_velocity(sender_id, step)
        transaction = rca.rule_checking_agent(transaction, blacklist=self.blacklist)
        transaction = da.make_decision(transaction, ml_score_threshold=self.current_threshold)
        transaction = aa.take_action(transaction)

        # Apply the real-world effect of the decision to account balances.
        if transaction["final_decision"] == "approve":
            self.storage.set_account_balance(
                sender["account_id"], sender["available_balance"] - amount, sender["held_balance"]
            )
            if receiver is not None:
                self.storage.set_account_balance(
                    receiver["account_id"], receiver["available_balance"] + amount, receiver["held_balance"]
                )
        elif transaction["final_decision"] == "flag":
            # Hold the funds: move them out of "available" but not yet
            # out of the account entirely, pending analyst review.
            self.storage.set_account_balance(
                sender["account_id"], sender["available_balance"] - amount, sender["held_balance"] + amount
            )
        # 'block' -> no balance change at all; funds never left the account.

        self.storage.log_live_transaction(transaction)
        return transaction

    def record_resolved_review(self, transaction_id, is_fraud):
        """
        Feeds an analyst's Confirm/Dismiss decision on a live
        transaction into the Learning Agent's pending batch, as real
        ground truth. Live customer transactions have no PaySim
        isFraud label the way historical rows do -- an analyst's
        resolution IS the ground truth for these, and without this
        method the Learning Agent would never see live traffic at
        all, no matter how many reviews get resolved.

        Returns a small dict describing what happened: whether this
        push triggered an actual learning pass, and the batch's
        current size either way, so the admin dashboard can show
        "3 of 5 resolutions until the next automatic threshold
        review" as a concrete, honest number instead of a vague
        "learning in progress" claim.
        """
        txn = self.storage.get_live_transaction(transaction_id)
        if txn is None:
            raise ValueError(f"No live transaction found with id {transaction_id}")

        self._pending_batch.append({
            "transaction_id": txn["transaction_id"],
            "amount": txn["amount"],
            "ml_score": txn["ml_score"],
            "rules_verdict": txn["rules_verdict"],
            "final_decision": txn["final_decision"],
            "actual_fraud": is_fraud,
        })

        if len(self._pending_batch) >= self.learn_every:
            report = self._run_learning_pass()
            return {"learning_pass_triggered": True, "report": report, "batch_size": 0, "learn_every": self.learn_every}

        return {
            "learning_pass_triggered": False,
            "report": None,
            "batch_size": len(self._pending_batch),
            "learn_every": self.learn_every,
        }

    def _record_and_count_velocity(self, sender_id, step):
        """
        Records this transaction in the global sliding window and
        returns how many transactions (including this one) that
        sender has made within the last `velocity_window_steps`.
        Entries older than the window are evicted from the FRONT of
        the window as new ones arrive at the back -- since PaySim's
        CSV is in non-decreasing step order, this keeps the window
        (and therefore memory use) bounded by transaction volume
        within the window, not by total unique senders in the file.
        """
        self._velocity_window.append((step, sender_id))
        self._velocity_counts[sender_id] += 1

        while self._velocity_window and (step - self._velocity_window[0][0]) > self.velocity_window_steps:
            old_step, old_sender = self._velocity_window.popleft()
            self._velocity_counts[old_sender] -= 1
            if self._velocity_counts[old_sender] <= 0:
                del self._velocity_counts[old_sender]

        return self._velocity_counts[sender_id]

    def _run_learning_pass(self):
        """Runs the Learning Agent on the pending batch and updates
        the threshold used for future decisions."""
        report = la.learning_agent(self._pending_batch, self.current_threshold)
        report["run_at"] = datetime.now().isoformat(timespec="seconds")
        report["batch_size"] = len(self._pending_batch)

        self.learning_history.append(report)
        self.current_threshold = report["new_threshold"]
        if self.storage:
            self.storage.set_state("current_threshold", self.current_threshold)

        self._pending_batch = []
        return report

    def flush(self):
        """
        Force a Learning Agent pass on whatever is left in the
        pending batch, even if it hasn't reached learn_every yet.
        Call this at the end of a run (e.g. end of CSV file) so no
        transactions are silently dropped from evaluation.
        """
        if self._pending_batch:
            return self._run_learning_pass()
        return None

    def run_on_csv(self, csv_path, chunksize=100_000, max_rows=None, verbose=True):
        """
        Runs the pipeline over a PaySim CSV that's too large to load
        into memory at once (the full PaySim file is ~6.3M rows).
        Reads it in chunks, batch-scoring each chunk's ML predictions
        in one call (see process_rows), and calls flush() after each
        chunk so learning passes still happen at learn_every.

        max_rows: stop after this many rows total (handy for a quick
        end-to-end test run before committing to the full file).
        """
        import pandas as pd
        import time

        rows_done = 0
        row_index = 0
        start_time = time.time()
        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                records = chunk.to_dict("records")
                if max_rows is not None and rows_done + len(records) > max_rows:
                    records = records[: max_rows - rows_done]

                self.process_rows(records, row_index)
                row_index += len(records)
                rows_done += len(records)

                # Flush after every chunk, not just at the end, so
                # progress already made is safely on disk if the run
                # gets interrupted (sleep, Ctrl+C, power loss, etc.).
                if self.storage:
                    self.storage.flush()

                if verbose:
                    elapsed = time.time() - start_time
                    rate = rows_done / elapsed if elapsed > 0 else 0
                    print(f"Processed {rows_done} rows so far... "
                          f"(threshold now {self.current_threshold}, "
                          f"{rate:.0f} rows/sec, {elapsed:.1f}s elapsed)")

                if max_rows is not None and rows_done >= max_rows:
                    break
        except KeyboardInterrupt:
            print(f"\nInterrupted after {rows_done} rows. Saving progress...")
            self.flush()
            if self.storage:
                self.storage.flush()
            print("Progress saved. Partial results are in self.summary() "
                  "and whatever db_path you gave the Orchestrator.")
            return self.summary()

        self.flush()
        if self.storage:
            self.storage.flush()
        if verbose and max_rows is not None and rows_done >= max_rows:
            print(f"Stopped at {rows_done} rows (max_rows).")
        return self.summary()

    def run_on_dataframe(self, df):
        """
        Convenience method: runs every row of a pandas DataFrame
        (loaded from the PaySim CSV) through the pipeline, using the
        same batched ML scoring as run_on_csv.
        Returns (results, quarantined, learning_history).
        """
        results = self.process_rows(df.to_dict("records"), 0)
        self.flush()
        return results, self.quarantined, self.learning_history

    def summary(self):
        """Quick counts for a sanity check / demo printout."""
        total = len(self.quarantined) + len(self._pending_batch)
        # Note: decided transactions already folded into learning
        # passes aren't in _pending_batch any more, so this summary
        # is most meaningful right after flush() has NOT yet cleared
        # things, or by reading action_agent.get_action_log().
        return {
            "quarantined": len(self.quarantined),
            "pending_for_learning": len(self._pending_batch),
            "learning_passes_run": len(self.learning_history),
            "current_threshold": self.current_threshold,
            "action_log_entries": len(aa.get_action_log()),
        }


if __name__ == "__main__":
    import json

    # Small smoke test using hand-built PaySim-shaped rows (no CSV
    # needed). Includes one dirty row to exercise quarantine, and a
    # known real fraud row from the PaySim TRANSFER pattern.
    demo_rows = [
        {  # normal payment
            "step": 1, "type": "PAYMENT", "amount": 9839.64,
            "nameOrig": "C1231006815", "oldbalanceOrg": 170136.0, "newbalanceOrig": 160296.36,
            "nameDest": "M1979787155", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "isFraud": 0,
        },
        {  # known real fraud pattern: full account drain via TRANSFER
            "step": 1, "type": "TRANSFER", "amount": 181.0,
            "nameOrig": "C1305486145", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
            "nameDest": "C553264065", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "isFraud": 1,
        },
        {  # dirty row: negative amount should fail cleaning
            "step": 5, "type": "CASH_OUT", "amount": -500.0,
            "nameOrig": "C9999999999", "oldbalanceOrg": 500.0, "newbalanceOrig": 1000.0,
            "nameDest": "C8888888888", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
            "isFraud": 0,
        },
    ]

    orch = Orchestrator(blacklist={"C8888888888"}, learn_every=2)

    print("=== PROCESSING ROWS ===")
    for i, row in enumerate(demo_rows):
        result = orch.process_row(row, i)
        print(json.dumps(result, indent=2, default=str))

    orch.flush()

    print("\n=== SUMMARY ===")
    print(json.dumps(orch.summary(), indent=2))

    print("\n=== LEARNING HISTORY ===")
    print(json.dumps(orch.learning_history, indent=2))
