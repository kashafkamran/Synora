"""
US-03: Server Endpoint — Serve Preprocessed Shards to Browser Clients
Synora — Browser-Based Federated Learning Toolkit

Adds to your Flask coordination server:
  GET  /data/preprocessed/client_<id>_preprocessed.json   ← client fetches its shard
  GET  /data/preprocessed/vocabulary.json                  ← optional: full vocab
  POST /api/preprocess/run                                 ← trigger preprocessing
  GET  /api/preprocess/status                              ← check if preprocessing done

Integrate this into your main server.py with:
    from preprocessing_server import preprocessing_bp
    app.register_blueprint(preprocessing_bp)
"""

import json
import os
from pathlib import Path
from flask import Blueprint, jsonify, send_file, request, abort

# ── Import the preprocessing pipeline ──────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from preprocessing import run_preprocessing_pipeline, OUTPUT_DIR


preprocessing_bp = Blueprint("preprocessing", __name__)

# ─────────────────────────────────────────────
#  ROUTE 1: Serve a client's preprocessed shard
# ─────────────────────────────────────────────

@preprocessing_bp.route(
    "/data/preprocessed/client_<int:client_id>_preprocessed.json",
    methods=["GET"]
)
def serve_client_shard(client_id: int):
    """
    Browser clients call this at the start of each training round
    to load their preprocessed data shard.

    Used by preprocessing_client.js → loadClientShard()
    """
    shard_path = OUTPUT_DIR / f"client_{client_id}_preprocessed.json"

    if not shard_path.exists():
        abort(404, description=f"Preprocessed shard for client {client_id} not found. "
                               f"Run preprocessing first via POST /api/preprocess/run")

    return send_file(
        shard_path,
        mimetype="application/json",
        as_attachment=False
    )


# ─────────────────────────────────────────────
#  ROUTE 2: Serve the vocabulary file
# ─────────────────────────────────────────────

@preprocessing_bp.route("/data/preprocessed/vocabulary.json", methods=["GET"])
def serve_vocabulary():
    """
    Returns the shared vocabulary JSON.
    Clients can use this to inspect the vocab or for debugging.
    """
    vocab_path = OUTPUT_DIR / "vocabulary.json"

    if not vocab_path.exists():
        abort(404, description="Vocabulary not built yet. Run preprocessing first.")

    return send_file(vocab_path, mimetype="application/json")


# ─────────────────────────────────────────────
#  ROUTE 3: Trigger preprocessing pipeline
# ─────────────────────────────────────────────

@preprocessing_bp.route("/api/preprocess/run", methods=["POST"])
def run_preprocessing():
    """
    Triggers the full US-03 preprocessing pipeline on the current
    partitioned shards (output of US-02).

    Expected JSON body:
    {
      "shards": {
        "0": [{"text": "Habari yako", "label": "greeting"}, ...],
        "1": [...],
        ...
      }
    }

    Returns:
    {
      "status": "success",
      "vocab_size": 342,
      "sequence_length": 50,
      "num_clients": 3,
      "output_paths": {"0": "...", "1": "...", "2": "..."}
    }
    """
    body = request.get_json(silent=True)
    if not body or "shards" not in body:
        abort(400, description='Request body must contain "shards" key with client data.')

    # Convert string keys to int keys (JSON keys are always strings)
    try:
        shards = {int(k): v for k, v in body["shards"].items()}
    except (ValueError, AttributeError) as e:
        abort(400, description=f"Invalid shards format: {e}")

    if not shards:
        abort(400, description="Shards dict is empty.")

    try:
        result = run_preprocessing_pipeline(shards)
    except ValueError as e:
        # Validation errors from the pipeline
        abort(422, description=str(e))
    except Exception as e:
        abort(500, description=f"Preprocessing pipeline error: {e}")

    return jsonify({
        "status": "success",
        "vocab_size": result["vocab_size"],
        "sequence_length": result["sequence_length"],
        "label_map": result["label_map"],
        "num_clients": len(result["output_paths"]),
        "output_paths": result["output_paths"]
    }), 200


# ─────────────────────────────────────────────
#  ROUTE 4: Check preprocessing status
# ─────────────────────────────────────────────

@preprocessing_bp.route("/api/preprocess/status", methods=["GET"])
def preprocessing_status():
    """
    Returns whether preprocessing has been run and which client shards exist.

    Response:
    {
      "preprocessed": true,
      "vocab_exists": true,
      "clients": [0, 1, 2],
      "sequence_length": 50,
      "vocab_size": 342
    }
    """
    vocab_path = OUTPUT_DIR / "vocabulary.json"
    vocab_exists = vocab_path.exists()

    # Find all shard files
    clients = []
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("client_*_preprocessed.json"):
            try:
                client_id = int(f.stem.split("_")[1])
                clients.append(client_id)
            except (ValueError, IndexError):
                pass
    clients.sort()

    # Read vocab size if available
    vocab_size = 0
    sequence_length = 0
    if vocab_exists:
        with open(vocab_path) as f:
            vocab = json.load(f)
        vocab_size = len(vocab)

    # Read sequence length from first shard if available
    if clients:
        first_shard = OUTPUT_DIR / f"client_{clients[0]}_preprocessed.json"
        with open(first_shard) as f:
            shard_data = json.load(f)
        sequence_length = shard_data.get("sequence_length", 0)

    return jsonify({
        "preprocessed": vocab_exists and len(clients) > 0,
        "vocab_exists": vocab_exists,
        "clients": clients,
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
    }), 200
