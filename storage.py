"""
storage.py
==========
Persistent storage for the fraud pipeline, per Section 4 of the
proposal ("Storage: SQLite -- Free, file-based, zero setup -- logs
every transaction & verdict"). This is what the Streamlit dashboard
will read from later.

Writes are buffered in memory and flushed to disk in batches, since
committing on every single row would be far too slow across millions
of PaySim transactions.
"""

import json
import sqlite3
import threading

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id          TEXT PRIMARY KEY,
    sender_id               TEXT,
    receiver_id             TEXT,
    receiver_type           TEXT,
    amount                  REAL,
    sender_balance_before   REAL,
    timestamp               TEXT,
    is_clean                INTEGER,
    ml_score                REAL,
    rules_triggered         TEXT,
    rules_verdict           TEXT,
    final_decision          TEXT,
    decision_reason         TEXT,
    action_taken            TEXT,
    transaction_status      TEXT,
    alert_raised            INTEGER,
    action_message          TEXT,
    action_timestamp        TEXT,
    actual_fraud            INTEGER
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO transactions (
    transaction_id, sender_id, receiver_id, receiver_type, amount,
    sender_balance_before, timestamp, is_clean, ml_score,
    rules_triggered, rules_verdict, final_decision, decision_reason,
    action_taken, transaction_status, alert_raised, action_message,
    action_timestamp, actual_fraud
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# The human-in-the-loop queue: the proposal (Section 3) names an
# "optional seventh role -- the Human Analyst" for cases the Decision
# Agent can't resolve with confidence. In this system that's every
# 'flag' verdict: not clear enough to approve, not certain enough to
# block outright. An analyst resolving one of these is exactly the
# kind of new ground-truth label the Learning Agent is designed to
# consume on its next pass.
CREATE_REVIEW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS review_queue (
    transaction_id   TEXT PRIMARY KEY,
    sender_id        TEXT,
    receiver_id      TEXT,
    amount           REAL,
    timestamp        TEXT,
    ml_score         REAL,
    rules_triggered  TEXT,
    decision_reason  TEXT,
    status           TEXT DEFAULT 'pending',
    queued_at        TEXT,
    resolved_at      TEXT
)
"""


REVIEW_INSERT_SQL = """
INSERT OR REPLACE INTO review_queue (
    transaction_id, sender_id, receiver_id, amount, timestamp,
    ml_score, rules_triggered, decision_reason, status, queued_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
"""

# ---------------------------------------------------------------------
# Virtual bank layer: demo customer accounts, live/interactive
# transactions, and shared pipeline state. Kept in separate tables
# from `transactions` (the historical evaluation log) on purpose --
# the millions of PaySim rows used for Chapter 4's metrics must never
# mix with made-up demo transactions that have no real ground truth.
# ---------------------------------------------------------------------

CREATE_ACCOUNTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id         TEXT PRIMARY KEY,
    display_name       TEXT,
    pin                TEXT,
    bank_name          TEXT,
    available_balance  REAL,
    held_balance        REAL DEFAULT 0
)
"""

# Analyst logins for the Fraud Ops Command Center. Kept as plain text
# for this coursework demo, same as the customer PINs above -- a real
# deployment would hash these (e.g. bcrypt) and is exactly the kind
# of thing worth flagging in the report's data-privacy discussion.
CREATE_ADMIN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS admin_users (
    username      TEXT PRIMARY KEY,
    password      TEXT,
    display_name  TEXT
)
"""

# available_balance: funds the customer can actually spend right now.
# held_balance: funds reserved against a 'flag'-verdict transaction
# that's awaiting analyst review -- real banks do exactly this rather
# than either completing or fully blocking an uncertain transaction.
CREATE_LIVE_TXN_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_transactions (
    transaction_id        TEXT PRIMARY KEY,
    sender_id             TEXT,
    receiver_id           TEXT,
    receiver_type         TEXT,
    type                  TEXT,
    amount                REAL,
    sender_balance_before REAL,
    timestamp             TEXT,
    ml_score              REAL,
    rules_triggered       TEXT,
    rules_verdict         TEXT,
    final_decision        TEXT,
    decision_reason       TEXT,
    action_taken          TEXT,
    transaction_status    TEXT,
    resolution            TEXT,
    created_at            TEXT,
    resolved_at           TEXT
)
"""

# Small key-value store so state that needs to be shared across
# separate processes (the customer app and the admin dashboard each
# run as their own `streamlit run` process, with no shared Python
# memory) lives in one place both can read and write. Used for the
# live ML threshold and a simulated "current step" clock.
CREATE_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_state (
    key    TEXT PRIMARY KEY,
    value  TEXT
)
"""


class PipelineStorage:
    def __init__(self, db_path="fraud_pipeline.db", buffer_size=1000):
        # check_same_thread=False: Flask's dev server handles each
        # request on its own thread, but this connection is created
        # once at startup on the main thread. Without this, SQLite
        # refuses to let any other thread touch it. A single global
        # lock (below) still serializes actual access, since SQLite
        # connections aren't safe for concurrent use even when this
        # check is turned off.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()  # reentrant: some methods below call each other
        # Trade a little durability for a lot of write speed -- fine
        # for a coursework demo, not something to do in production.
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = MEMORY")
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.execute(CREATE_REVIEW_TABLE_SQL)
        self.conn.execute(CREATE_ACCOUNTS_TABLE_SQL)
        self.conn.execute(CREATE_ADMIN_TABLE_SQL)
        self.conn.execute(CREATE_LIVE_TXN_TABLE_SQL)
        self.conn.execute(CREATE_STATE_TABLE_SQL)
        self.conn.commit()

        self.buffer_size = buffer_size
        self._buffer = []
        self._review_buffer = []

    def log(self, transaction, actual_fraud=None):
        """
        Queues one transaction dict for writing. Works for both fully
        decided transactions and quarantined (is_clean=False) ones --
        missing fields just come through as NULL.
        """
        rules_triggered = transaction.get("rules_triggered")
        row = (
            transaction.get("transaction_id"),
            transaction.get("sender_id"),
            transaction.get("receiver_id"),
            transaction.get("receiver_type"),
            transaction.get("amount"),
            transaction.get("sender_balance_before"),
            transaction.get("timestamp"),
            int(bool(transaction.get("is_clean"))) if "is_clean" in transaction else None,
            transaction.get("ml_score"),
            json.dumps(rules_triggered) if rules_triggered is not None else None,
            transaction.get("rules_verdict"),
            transaction.get("final_decision"),
            transaction.get("decision_reason"),
            transaction.get("action_taken"),
            transaction.get("transaction_status"),
            int(bool(transaction.get("alert_raised"))) if "alert_raised" in transaction else None,
            transaction.get("action_message"),
            transaction.get("action_timestamp"),
            int(bool(actual_fraud)) if actual_fraud is not None else None,
        )
        self._buffer.append(row)

        if len(self._buffer) >= self.buffer_size:
            with self._lock:
                self.flush()

    def queue_for_review(self, transaction):
        """
        Adds a 'flag'-verdict transaction to the human-analyst review
        queue. Buffered the same way as log(), rather than committed
        individually -- at real dataset scale, flagged transactions
        can number in the hundreds of thousands, and one disk commit
        per row adds up to a serious, easy-to-miss slowdown.
        """
        import datetime as _dt

        row = (
            transaction.get("transaction_id"),
            transaction.get("sender_id"),
            transaction.get("receiver_id"),
            transaction.get("amount"),
            transaction.get("timestamp"),
            transaction.get("ml_score"),
            json.dumps(transaction.get("rules_triggered")),
            transaction.get("decision_reason"),
            _dt.datetime.now().isoformat(timespec="seconds"),
        )
        self._review_buffer.append(row)

        if len(self._review_buffer) >= self.buffer_size:
            with self._lock:
                self.flush()

    def list_pending_reviews(self):
        """Returns all transactions still awaiting analyst decision."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM review_queue WHERE status = 'pending' ORDER BY queued_at"
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def resolve_review(self, transaction_id, is_fraud):
        """
        Records an analyst's decision on a flagged transaction. This
        is the human-in-the-loop path named in the proposal (Section
        3): the analyst's confirmed True/False becomes a new
        ground-truth label the Learning Agent can use, feeding the
        loop shown in Section 3.1's diagram back into the system.

        This one IS written immediately, not buffered -- an analyst
        resolving a single case is a rare, deliberate action (not a
        per-transaction bulk write), and its own dashboard should see
        the update right away.
        """
        import datetime as _dt

        status = "confirmed_fraud" if is_fraud else "confirmed_legitimate"
        with self._lock:
            self.conn.execute(
                "UPDATE review_queue SET status = ?, resolved_at = ? WHERE transaction_id = ?",
                (status, _dt.datetime.now().isoformat(timespec="seconds"), transaction_id),
            )
            self.conn.commit()

    def flush(self):
        """Writes any buffered rows (both tables) to disk. Safe to call anytime."""
        with self._lock:
            if self._buffer:
                self.conn.executemany(INSERT_SQL, self._buffer)
                self._buffer.clear()
            if self._review_buffer:
                self.conn.executemany(REVIEW_INSERT_SQL, self._review_buffer)
                self._review_buffer.clear()
            self.conn.commit()

    def close(self):
        self.flush()
        with self._lock:
            self.conn.close()

    # -------------------------------------------------------------
    # Virtual bank: accounts
    # -------------------------------------------------------------
    def seed_accounts(self, accounts):
        """
        Inserts each account (a list of {account_id, display_name,
        pin, bank_name, available_balance} dicts) only if it doesn't
        already exist, so re-running this on every app startup
        doesn't reset balances mid-demo.
        """
        for acct in accounts:
            with self._lock:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO accounts
                        (account_id, display_name, pin, bank_name, available_balance, held_balance)
                    VALUES (?, ?, ?, ?, ?, 0)
                    """,
                    (
                        acct["account_id"],
                        acct["display_name"],
                        acct["pin"],
                        acct["bank_name"],
                        acct["available_balance"],
                    ),
                )
                self.conn.commit()

    def get_account(self, account_id):
        with self._lock:
            cur = self.conn.execute(
                "SELECT account_id, display_name, bank_name, available_balance, held_balance "
                "FROM accounts WHERE account_id = ?",
                (account_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "account_id": row[0],
            "display_name": row[1],
            "bank_name": row[2],
            "available_balance": row[3],
            "held_balance": row[4],
        }

    def list_accounts(self):
        with self._lock:
            cur = self.conn.execute(
                "SELECT account_id, display_name, bank_name, available_balance, held_balance "
                "FROM accounts ORDER BY account_id"
            )
            cols = ["account_id", "display_name", "bank_name", "available_balance", "held_balance"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def verify_login(self, account_id, pin):
        """
        Checks account_id + pin together, the way a real bank login
        would -- returns the account (never the pin itself) on
        success, or None on any mismatch. Deliberately doesn't
        distinguish "wrong account" from "wrong pin" in what it
        returns, same as a real login should never reveal which part
        was wrong.
        """
        with self._lock:
            cur = self.conn.execute(
                "SELECT account_id, display_name, bank_name, available_balance, held_balance "
                "FROM accounts WHERE account_id = ? AND pin = ?",
                (account_id, pin),
            )
            row = cur.fetchone()
        if row is None:
            return None
        cols = ["account_id", "display_name", "bank_name", "available_balance", "held_balance"]
        return dict(zip(cols, row))

    def seed_admin_users(self, admins):
        """
        Inserts each analyst login (a list of {username, password,
        display_name} dicts) only if it doesn't already exist, same
        pattern as seed_accounts.
        """
        for admin in admins:
            with self._lock:
                self.conn.execute(
                    "INSERT OR IGNORE INTO admin_users (username, password, display_name) VALUES (?, ?, ?)",
                    (admin["username"], admin["password"], admin["display_name"]),
                )
                self.conn.commit()

    def verify_admin_login(self, username, password):
        """
        Same shape as the customer verify_login: checks username +
        password together, returns the analyst's info (never the
        password) on success, None on any mismatch.
        """
        with self._lock:
            cur = self.conn.execute(
                "SELECT username, display_name FROM admin_users WHERE username = ? AND password = ?",
                (username, password),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"username": row[0], "display_name": row[1]}

    def set_account_balance(self, account_id, available_balance, held_balance):
        """Writes both balances immediately -- account updates are rare,
        low-volume events (one customer action at a time), not a bulk
        stream, so there's no need to buffer these."""
        with self._lock:
            self.conn.execute(
                "UPDATE accounts SET available_balance = ?, held_balance = ? WHERE account_id = ?",
                (available_balance, held_balance, account_id),
            )
            self.conn.commit()

    # -------------------------------------------------------------
    # Virtual bank: live/interactive transactions
    # -------------------------------------------------------------
    def log_live_transaction(self, transaction):
        """
        Writes one interactive (customer-app-submitted) transaction.
        Written immediately, not buffered -- these are one-at-a-time
        customer actions, not a bulk historical stream, and the admin
        dashboard needs to see each one right away.
        """
        import datetime as _dt

        rules_triggered = transaction.get("rules_triggered")
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO live_transactions (
                    transaction_id, sender_id, receiver_id, receiver_type, type,
                    amount, sender_balance_before, timestamp, ml_score,
                    rules_triggered, rules_verdict, final_decision, decision_reason,
                    action_taken, transaction_status, resolution, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    transaction.get("transaction_id"),
                    transaction.get("sender_id"),
                    transaction.get("receiver_id"),
                    transaction.get("receiver_type"),
                    transaction.get("type"),
                    transaction.get("amount"),
                    transaction.get("sender_balance_before"),
                    transaction.get("timestamp"),
                    transaction.get("ml_score"),
                    json.dumps(rules_triggered) if rules_triggered is not None else None,
                    transaction.get("rules_verdict"),
                    transaction.get("final_decision"),
                    transaction.get("decision_reason"),
                    transaction.get("action_taken"),
                    transaction.get("transaction_status"),
                    _dt.datetime.now().isoformat(timespec="seconds"),
                ),
            )
            self.conn.commit()

    def list_live_transactions(self, limit=50):
        """Most recent live transactions first, for the admin feed."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM live_transactions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_live_transaction(self, transaction_id):
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM live_transactions WHERE transaction_id = ?", (transaction_id,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [c[0] for c in cur.description]
            return dict(zip(cols, row))

    def list_pending_live_reviews(self):
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM live_transactions "
                "WHERE transaction_status = 'pending_review' AND resolution IS NULL "
                "ORDER BY created_at"
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def resolve_live_transaction(self, transaction_id, is_fraud):
        """
        The admin dashboard's Confirm/Dismiss action on a held
        transaction. This is where the hold actually gets settled:

          - is_fraud=True  -> the transaction is cancelled. Held
            funds are released back to the sender's available
            balance; nothing is sent to the receiver.
          - is_fraud=False -> the transaction goes ahead. Held funds
            leave the sender's held balance permanently; if the
            receiver is also a demo account, they're credited.

        Mirrors a real bank's hold-then-settle flow for a transaction
        that was neither clearly fine nor clearly fraud on its own.
        """
        import datetime as _dt

        txn = self.get_live_transaction(transaction_id)
        if txn is None:
            raise ValueError(f"No live transaction found with id {transaction_id}")
        if txn["transaction_status"] != "pending_review" or txn["resolution"] is not None:
            raise ValueError(f"Transaction {transaction_id} is not awaiting review")

        amount = txn["amount"]
        sender = self.get_account(txn["sender_id"])
        if sender is None:
            raise ValueError(f"Sender account {txn['sender_id']} not found")

        if is_fraud:
            new_available = sender["available_balance"] + amount
            new_held = sender["held_balance"] - amount
            self.set_account_balance(sender["account_id"], new_available, new_held)
            new_status, resolution = "blocked", "confirmed_fraud"
        else:
            new_held = sender["held_balance"] - amount
            self.set_account_balance(sender["account_id"], sender["available_balance"], new_held)
            receiver = self.get_account(txn["receiver_id"])
            if receiver is not None:
                self.set_account_balance(
                    receiver["account_id"],
                    receiver["available_balance"] + amount,
                    receiver["held_balance"],
                )
            new_status, resolution = "approved", "confirmed_legitimate"

        with self._lock:
            self.conn.execute(
                "UPDATE live_transactions SET transaction_status = ?, resolution = ?, resolved_at = ? "
                "WHERE transaction_id = ?",
                (new_status, resolution, _dt.datetime.now().isoformat(timespec="seconds"), transaction_id),
            )
            self.conn.commit()
        return new_status

    # -------------------------------------------------------------
    # Shared pipeline state (so separate Streamlit processes agree
    # on the current threshold and simulated clock)
    # -------------------------------------------------------------
    def get_state(self, key, default=None):
        with self._lock:
            cur = self.conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row is not None else default

    def set_state(self, key, value):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO pipeline_state (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            self.conn.commit()
