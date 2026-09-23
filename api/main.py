"""
api/main.py
===========
Flask backend for the virtual-bank demo. This is the ONLY place the
frontend (customer app + admin dashboard) talks to -- it never touches
orchestrator.py or storage.py directly. Run it from the fraud_pipeline
folder (not from inside api/) so the six agent modules import cleanly:

    cd fraud_pipeline
    python api/main.py

Serves:
  - JSON endpoints under /api/...
  - the frontend/ folder as static files, so opening
    http://127.0.0.1:8000/ loads the landing page directly.

This keeps the historical evaluation database (fraud_pipeline_full.db,
built from the 5.1M-row PaySim run) completely untouched -- the live
demo writes to its own separate bank_demo.db.
"""

import os
import random
import sys
import time

# Allow "import orchestrator" etc. to work when this file is run
# from inside api/, by adding the parent (fraud_pipeline) folder to
# the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory

from orchestrator import Orchestrator, seed_demo_accounts, seed_demo_admins, build_blacklist_from_paysim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
BANK_DB_PATH = os.path.join(BASE_DIR, "bank_demo.db")
PAYSIM_CSV_PATH = os.path.join(BASE_DIR, "paysim.csv")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# ---------------------------------------------------------------------
# One shared Orchestrator for the life of the server process. Flask's
# dev server is single-process by default, so this is safe -- every
# request handler below uses this same instance and its one
# storage/SQLite connection.
# ---------------------------------------------------------------------
_blacklist = set()
if os.path.exists(PAYSIM_CSV_PATH):
    try:
        _blacklist = build_blacklist_from_paysim(PAYSIM_CSV_PATH, min_fraud_count=2)
    except Exception as exc:  # pragma: no cover - startup diagnostics only
        print(f"Could not build blacklist from {PAYSIM_CSV_PATH}: {exc}")
else:
    print(f"No paysim.csv found at {PAYSIM_CSV_PATH} -- starting with an empty blacklist.")

# learn_every=5 here (vs. 500 for the historical CSV evaluation) --
# a live demo will only produce a handful of analyst-resolved
# transactions, so a small batch size is what makes the Learning
# Agent's threshold updates actually visible during a demo instead
# of requiring hundreds of resolutions first.
orch = Orchestrator(blacklist=_blacklist, db_path=BANK_DB_PATH, learn_every=5)
seed_demo_accounts(orch.storage)
seed_demo_admins(orch.storage)


# ---------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# No custom catch-all route needed here: Flask's automatic static
# handling (static_folder=FRONTEND_DIR, static_url_path="" above)
# already serves every other file in frontend/ -- e.g. styles.css,
# customer.html, admin.html -- directly at the root URL. A second,
# identical "/<path:filename>" route would just conflict with it.


# ---------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------
@app.route("/api/accounts", methods=["GET"])
def list_accounts():
    return jsonify(orch.storage.list_accounts())


@app.route("/api/accounts/<account_id>", methods=["GET"])
def get_account(account_id):
    account = orch.storage.get_account(account_id)
    if account is None:
        return jsonify({"error": "not_found", "message": f"No account {account_id}"}), 404
    return jsonify(account)


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True) or {}
    account_id = data.get("account_id", "").strip().upper()
    pin = data.get("pin", "").strip()
    if not account_id or not pin:
        return jsonify({"error": "missing_fields", "message": "Account number and PIN are required."}), 400

    account = orch.storage.verify_login(account_id, pin)
    if account is None:
        return jsonify({"error": "invalid_credentials", "message": "Account number or PIN is incorrect."}), 401
    return jsonify(account)


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "missing_fields", "message": "Username and password are required."}), 400

    admin = orch.storage.verify_admin_login(username, password)
    if admin is None:
        return jsonify({"error": "invalid_credentials", "message": "Username or password is incorrect."}), 401
    return jsonify(admin)


# ---------------------------------------------------------------------
# Live transactions (customer app)
# ---------------------------------------------------------------------
@app.route("/api/transactions", methods=["POST"])
def submit_transaction():
    data = request.get_json(force=True) or {}
    required = ["sender_id", "receiver_id", "type", "amount"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": "missing_fields", "message": f"Missing: {missing}"}), 400

    result = orch.process_live_transaction(
        sender_id=data["sender_id"],
        receiver_id=data["receiver_id"],
        txn_type=data["type"],
        amount=float(data["amount"]),
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/live-transactions", methods=["GET"])
def list_live_transactions():
    limit = int(request.args.get("limit", 50))
    return jsonify(orch.storage.list_live_transactions(limit=limit))


# ---------------------------------------------------------------------
# Analyst review queue (admin dashboard)
# ---------------------------------------------------------------------
@app.route("/api/reviews", methods=["GET"])
def list_reviews():
    return jsonify(orch.storage.list_pending_live_reviews())


@app.route("/api/reviews/<transaction_id>/confirm-fraud", methods=["POST"])
def confirm_fraud(transaction_id):
    """
    Blocking a transaction the bank has judged fraudulent doesn't
    need the customer's consent -- this is the bank protecting them,
    the same way a real fraud team blocks a card unilaterally.
    Resolves immediately, no OTP step.
    """
    try:
        new_status = orch.storage.resolve_live_transaction(transaction_id, True)
        learning_update = orch.record_resolved_review(transaction_id, True)
    except ValueError as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400

    return jsonify({
        "transaction_id": transaction_id,
        "new_status": new_status,
        "learning_update": learning_update,
    })


# In-memory OTP store: {transaction_id: {"code": str, "is_fraud": bool, "expires_at": float}}
# This is a demo-only stand-in for a real SMS/email OTP gateway --
# there's nowhere to actually deliver a text message here, so the
# code is returned directly in the request-otp response, clearly
# labeled in the frontend as a development shortcut, not something a
# real production system would ever expose over the API.
#
# Only ever used for the Dismiss path now: releasing held funds back
# to the customer's control is the one action that should require
# their own confirmation, since it's their money moving.
_pending_otps = {}
OTP_TTL_SECONDS = 300


@app.route("/api/reviews/<transaction_id>/request-otp", methods=["POST"])
def request_review_otp(transaction_id):
    txn = orch.storage.get_live_transaction(transaction_id)
    if txn is None:
        return jsonify({"error": "not_found", "message": f"No transaction {transaction_id}"}), 404
    if txn["transaction_status"] != "pending_review" or txn["resolution"] is not None:
        return jsonify({"error": "already_resolved", "message": "This transaction is no longer awaiting review."}), 400

    code = f"{random.randint(0, 999999):06d}"
    _pending_otps[transaction_id] = {
        "code": code,
        "is_fraud": False,  # request-otp is only ever the Dismiss path now
        "sender_id": txn["sender_id"],
        "expires_at": time.time() + OTP_TTL_SECONDS,
    }
    return jsonify({"otp": code, "expires_in_seconds": OTP_TTL_SECONDS})


@app.route("/api/reviews/<transaction_id>/verify-otp", methods=["POST"])
def verify_review_otp(transaction_id):
    data = request.get_json(force=True) or {}
    submitted = (data.get("otp") or "").strip()

    pending = _pending_otps.get(transaction_id)
    if pending is None:
        return jsonify({"error": "no_otp_requested", "message": "Request a new code and try again."}), 400
    if time.time() > pending["expires_at"]:
        del _pending_otps[transaction_id]
        return jsonify({"error": "otp_expired", "message": "That code has expired. Request a new one."}), 400
    if submitted != pending["code"]:
        return jsonify({"error": "invalid_otp", "message": "Incorrect code."}), 401

    is_fraud = pending["is_fraud"]
    del _pending_otps[transaction_id]

    try:
        new_status = orch.storage.resolve_live_transaction(transaction_id, is_fraud)
        learning_update = orch.record_resolved_review(transaction_id, is_fraud)
    except ValueError as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400

    return jsonify({
        "transaction_id": transaction_id,
        "new_status": new_status,
        "learning_update": learning_update,
    })


@app.route("/api/customer/pending-verification", methods=["GET"])
def customer_pending_verification():
    """
    Polled by the customer app to discover whether an analyst has
    requested their verification to release a held transaction (the
    Dismiss path). Returns at most one pending item -- a customer
    should never have more than one transaction held at a time in
    this demo's flow, but if they somehow did, the oldest wins.
    """
    account_id = request.args.get("account_id", "").strip().upper()
    if not account_id:
        return jsonify({"error": "missing_fields", "message": "Missing: account_id"}), 400

    now = time.time()
    for transaction_id, pending in list(_pending_otps.items()):
        if pending["expires_at"] < now:
            del _pending_otps[transaction_id]
            continue
        if pending["sender_id"] != account_id:
            continue
        return jsonify({
            "pending": True,
            "transaction_id": transaction_id,
            "otp": pending["code"],
            "expires_in_seconds": round(pending["expires_at"] - now),
        })

    return jsonify({"pending": False})


# ---------------------------------------------------------------------
# System state / KPIs (admin dashboard header)
# ---------------------------------------------------------------------
@app.route("/api/state", methods=["GET"])
def get_state():
    last_pass = orch.learning_history[-1] if orch.learning_history else None
    return jsonify({
        "current_threshold": orch.current_threshold,
        "blacklist_size": len(orch.blacklist),
        "pending_for_learning": len(orch._pending_batch),
        "learn_every": orch.learn_every,
        "last_learning_pass": last_pass,
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Counts by transaction_status, not final_decision -- final_decision
    is the pipeline's original verdict and never changes, but
    transaction_status updates the moment an analyst resolves a
    flagged transaction (pending_review -> approved or blocked). Using
    final_decision here would leave a confirmed-fraud transaction
    permanently miscounted as "flagged" even after it's been blocked.
    """
    txns = orch.storage.list_live_transactions(limit=100000)
    approved = sum(1 for t in txns if t["transaction_status"] == "approved")
    flagged = sum(1 for t in txns if t["transaction_status"] == "pending_review")
    blocked = sum(1 for t in txns if t["transaction_status"] == "blocked")
    blocked_value = sum(t["amount"] for t in txns if t["transaction_status"] == "blocked")
    return jsonify({
        "total": len(txns),
        "approved": approved,
        "flagged": flagged,
        "blocked": blocked,
        "blocked_value": round(blocked_value, 2),
    })


if __name__ == "__main__":
    # threaded=False: the orchestrator keeps in-memory state (the
    # velocity tracker, current threshold) that isn't lock-protected
    # the way storage.py's database access is. Single-threaded
    # request handling is the simplest way to guarantee two customer
    # actions can never corrupt that state by running at once -- fine
    # for a demo with one or two people clicking around, not something
    # to keep for real concurrent traffic.
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=False)
