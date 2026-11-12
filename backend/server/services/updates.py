"""
server/services/updates.py
==========================
Receiving one client's weights for the current round (SRS US-10), including
the server-side privacy enforcement layer (SDS 3.1.2.7).
"""

import numpy as np

from server import data_service, weight_codec
from server.api_errors import bad_request, conflict, forbidden
from server.database import db
from server.models import Client, ClientUpdate, Round, utcnow
from server.services.common import (
    client_partition,
    event,
    get_session,
    latest_round,
    safe_float,
    safe_int,
)
from server.services.rounds import process_round

ALLOWED_UPDATE_KEYS = {
    "weights", "round", "round_number", "roundNumber", "client_id", "clientId", "dataset_size", "datasetSize",
    "local_epochs", "localEpochs", "backend_used", "backendUsed", "payload_size_bytes", "payloadSizeBytes",
    "metrics", "train_seconds", "trainSeconds", "model_id", "modelId",
}
ALLOWED_METRIC_KEYS = {"accuracy", "loss", "val_accuracy", "val_loss", "train_accuracy", "train_loss", "f1", "precision", "recall"}


def _metric(metrics: dict, *keys: str):
    """First finite numeric value among `keys`, else None."""
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            return float(value)
    return None


def _check_privacy(data: dict) -> dict:
    """Only model weights, round identifiers and numeric metrics may be received - never raw text."""
    unexpected = set(data.keys()) - ALLOWED_UPDATE_KEYS
    if unexpected:
        raise bad_request(
            f"Unexpected fields: {sorted(unexpected)}. Only weight tensors, round identifiers and numeric metrics are permitted.",
            sorted(unexpected)[0],
            "privacy_violation",
        )
    metrics = data.get("metrics") or {}
    if not isinstance(metrics, dict) or set(metrics.keys()) - ALLOWED_METRIC_KEYS:
        raise bad_request("'metrics' may only contain numeric accuracy/loss values", "metrics", "privacy_violation")
    return metrics


def submit_update(client: Client, data: dict, payload_bytes: int = 0) -> dict:
    """Receive one client's weights for the current round; aggregate when the round is complete."""
    if not isinstance(data, dict):
        raise bad_request("JSON object body required")
    metrics = _check_privacy(data)

    sess = get_session(client.session_id)
    if sess is None or sess.status != "active":
        raise conflict("Session is not active", "session_not_active")
    rn = data.get("round") or data.get("round_number") or data.get("roundNumber")
    try:
        rn = int(rn)
    except (TypeError, ValueError):
        raise bad_request("'round' is required", "round")
    if rn != sess.current_round:
        raise conflict(f"Round {rn} is not the current round ({sess.current_round}).", "stale_round")
    rnd = latest_round(sess.session_id, rn)
    if rnd is None or rnd.state not in ("open", "collecting"):
        raise conflict("Round is no longer accepting updates", "round_closed")
    if client.client_id not in rnd.selected_list():
        raise forbidden("This client was not selected for the current round", "not_selected")
    if ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rn, client_id=client.client_id).first():
        raise conflict("An update for this round was already received", "duplicate_update")

    try:
        weights = weight_codec.decode_weights(data.get("weights"))
    except weight_codec.PayloadError as e:
        event(sess.session_id, client.client_id, "update_rejected", e.message)
        db.session.commit()
        raise bad_request(e.message, e.field, "invalid_weights")

    cfg = sess.config()
    partition_type, alpha = client_partition(client, cfg)
    _, y = data_service.client_shard(partition_type, client.partition_index or 0, client.language or "", alpha, cfg["seed"])
    upd = ClientUpdate(
        session_id=sess.session_id,
        round_number=rn,
        client_id=client.client_id,
        dataset_size=int(len(y)),
        weights_blob=weight_codec.pack_blob(weights),
        local_accuracy=_metric(metrics, "accuracy", "train_accuracy", "val_accuracy"),
        local_loss=_metric(metrics, "loss", "train_loss", "val_loss"),
        train_seconds=safe_float(data.get("train_seconds", data.get("trainSeconds")), 0, 36000),
        payload_bytes=int(payload_bytes or 0),
        backend=client.backend,
        local_epochs=safe_int(data.get("local_epochs", data.get("localEpochs")), 1, 50),
    )
    db.session.add(upd)
    try:
        Round.query.filter(Round.id == rnd.id, Round.state == "open").update(
            {"state": "collecting"}, synchronize_session=False
        )
        client.last_seen = utcnow()
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise conflict("An update for this round was already received", "duplicate_update")

    db.session.refresh(sess)
    rnd = latest_round(sess.session_id, rn)
    process_round(sess, rnd)
    db.session.refresh(sess)
    rnd = latest_round(sess.session_id, rn)
    received = ClientUpdate.query.filter_by(session_id=sess.session_id, round_number=rn).count()
    return {
        "status": "accepted",
        "message": "Update received",
        "round": rn,
        "round_state": rnd.state,
        "aggregated": rnd.state == "complete",
        "updates_received": received,
        "required": rnd.min_clients,
        "selected": len(rnd.selected_list()),
        "model_version": sess.model_version,
    }
