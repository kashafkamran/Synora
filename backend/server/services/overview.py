"""
server/services/overview.py
===========================
What the Results page and the exports read: the GLOBAL summary (all clients
combined) and, for one client, its OWN round history, graphs and metrics.
"""

import json
from typing import List, Optional

import numpy as np

from server import model_spec
from server.database import db
from server.models import (
    Client,
    ClientUpdate,
    ExperimentLog,
    ExperimentSession,
    iso,
    utcnow,
)
from server.services.common import (
    client_partition,
    latest_round,
    live_clients,
    planned_rounds,
    rounds_done_map,
    wanted,
)


def model_info() -> dict:
    """Static description of the model (size, layers) shown on the Results page."""
    return {
        "name": model_spec.ARCHITECTURE_NAME,
        "parameters": model_spec.total_parameters(),
        "size_bytes": model_spec.model_size_bytes(),
        "size_mb": round(model_spec.model_size_bytes() / 1e6, 3),
        "limit_mb": 5,
        "layers": [
            "Embedding(5000x32)", "GlobalAveragePooling1D", "Dense(16, relu)", "Dropout(0.3)", "Dense(3, softmax)"
        ],
    }


def session_overview(sess: Optional[ExperimentSession], client_id: Optional[str] = None) -> dict:
    """
    One call for the dashboards: global summary, participants, round history
    and - when `client_id` is given - that client's own results (`my`).
    """
    if sess is None:
        return {"session": None, "summary": None, "clients": [], "rounds": [], "model": model_info()}
    cfg = sess.config()
    live_ids = {c.client_id for c in live_clients(sess.session_id)}
    rnd = latest_round(sess.session_id, sess.current_round) if sess.current_round else None
    selected = set(rnd.selected_list()) if rnd else set()
    submitted = set()
    if rnd:
        submitted = {
            u.client_id
            for u in ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rnd.round_number).all()
        }
    done_map = rounds_done_map(sess.session_id)
    rows = Client.query.filter_by(session_id=sess.session_id).order_by(Client.partition_index).all()
    clients = [
        {
            **c.to_dict(),
            "rounds_done": done_map.get(c.client_id, 0),
            "rounds_wanted": wanted(c, cfg),
            "online": c.client_id in live_ids,
            "selected_this_round": c.client_id in selected,
            "submitted_this_round": c.client_id in submitted,
        }
        for c in rows
    ]
    logs = ExperimentLog.query.filter_by(session_id=sess.session_id).order_by(ExperimentLog.round_number).all()
    latest = logs[-1] if logs else None
    return {
        "session": sess.to_dict(),
        "summary": _global_summary(sess, cfg, rows, clients, logs, done_map),
        "current_round": (rnd.to_dict() if rnd else None),
        "received": len(submitted),
        "clients": clients,
        "rounds": [l.to_dict() for l in logs],
        "latest": latest.to_dict() if latest else None,
        "my": _my_results(sess, client_id, cfg, done_map, logs) if client_id else None,
        "partition_indicator": "non_iid" if any(client_partition(c, cfg)[0] != "iid" for c in rows) else "iid",
        "model": model_info(),
        "server_time": iso(utcnow()),
    }


def _global_summary(sess, cfg: dict, rows: List[Client], clients: List[dict], logs: List[ExperimentLog], done_map: dict) -> dict:
    """GLOBAL numbers: the aggregated model of ALL clients (any device), evaluated by the server."""
    latest = logs[-1] if logs else None
    contributed = db.session.query(ClientUpdate.client_id).filter_by(session_id=sess.session_id).distinct().count()
    return {
        "status": sess.status,
        "rounds_completed": len(logs),
        "client_rounds": int(sum(l.client_count or 0 for l in logs)),
        "client_rounds_planned": int(sum(
            max(done_map.get(c.client_id, 0), wanted(c, cfg) if c.status != "left" else 0) for c in rows
        )),
        "aggregation_rounds": len(logs),
        "total_rounds": planned_rounds(sess, rows, done_map),
        "current_round": sess.current_round,
        "accuracy": latest.accuracy if latest else None,
        "loss": latest.loss if latest else None,
        "f1": latest.f1_macro if latest else None,
        "precision": latest.precision_macro if latest else None,
        "recall": latest.recall_macro if latest else None,
        "clients_total": len(clients),
        "clients_online": sum(1 for c in clients if c["online"]),
        "clients_contributed": contributed,
        "clients_last_round": latest.client_count if latest else 0,
        "devices": sorted({c["backend"] for c in clients if c.get("backend")}),
    }


def _my_results(sess, client_id: str, cfg: dict, done_map: dict, logs: List[ExperimentLog]) -> Optional[dict]:
    """Everything that belongs to ONE client: its own round history, graphs and summary."""
    c = Client.query.filter_by(client_id=client_id, session_id=sess.session_id).first()
    if c is None:
        return None
    global_logs = {l.round_number: l for l in logs}
    updates = (
        ClientUpdate.query.filter_by(session_id=sess.session_id, client_id=client_id)
        .order_by(ClientUpdate.round_number).all()
    )
    rounds = []
    for i, u in enumerate(updates, start=1):
        g = global_logs.get(u.round_number)
        rounds.append({
            "my_round": i,
            "round": u.round_number,
            "local_accuracy": u.local_accuracy,
            "local_loss": u.local_loss,
            "train_seconds": u.train_seconds,
            "dataset_size": u.dataset_size,
            "backend": u.backend,
            "val_accuracy": u.val_accuracy,
            "per_class": json.loads(u.per_class_json) if u.per_class_json else None,
            "global_accuracy": g.accuracy if g else None,
            "global_loss": g.loss if g else None,
            "global_f1": g.f1_macro if g else None,
            "timestamp": iso(u.created_at),
        })
    accs = [r["local_accuracy"] for r in rounds if r["local_accuracy"] is not None]
    losses = [r["local_loss"] for r in rounds if r["local_loss"] is not None]
    secs = [r["train_seconds"] for r in rounds if r["train_seconds"] is not None]
    partition_type, alpha = client_partition(c, cfg)
    return {
        "client": {**c.to_dict(), "partition_type": partition_type, "dirichlet_alpha": alpha},
        "rounds": rounds,
        "summary": {
            "rounds_done": done_map.get(c.client_id, 0),
            "rounds_wanted": wanted(c, cfg),
            "joined_round": c.joined_round or 0,
            "samples": rounds[-1]["dataset_size"] if rounds else None,
            "last_local_accuracy": accs[-1] if accs else None,
            "best_local_accuracy": max(accs) if accs else None,
            "avg_local_accuracy": float(np.mean(accs)) if accs else None,
            "last_local_loss": losses[-1] if losses else None,
            "avg_train_seconds": float(np.mean(secs)) if secs else None,
            "total_train_seconds": float(np.sum(secs)) if secs else 0.0,
            "global_accuracy_at_last_round": rounds[-1]["global_accuracy"] if rounds else None,
        },
    }
