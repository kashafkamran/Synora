# SBFLT-19: Aggregation participation threshold - updates service

MIN_CLIENTS_THRESHOLD = 2  # minimum clients required before aggregation starts

def can_aggregate(connected_clients: int, threshold: int = MIN_CLIENTS_THRESHOLD) -> bool:
    """Check if enough clients are connected to start aggregation."""
    return connected_clients >= threshold

def get_threshold() -> int:
    """Return the current aggregation threshold value."""
    return MIN_CLIENTS_THRESHOLD

def set_threshold(new_threshold: int) -> dict:
    """Update the aggregation threshold value."""
    global MIN_CLIENTS_THRESHOLD
    if new_threshold < 1:
        return {"success": False, "message": "Threshold must be at least 1"}
    MIN_CLIENTS_THRESHOLD = new_threshold
    return {"success": True, "threshold": MIN_CLIENTS_THRESHOLD}
