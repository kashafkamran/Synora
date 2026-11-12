# SBFLT-19: Aggregation participation threshold - overview service

from backend.server.services.updates import get_threshold, can_aggregate

def get_server_overview(connected_clients: int) -> dict:
    """Return server status including aggregation threshold info."""
    threshold = get_threshold()
    return {
        "connected_clients": connected_clients,
        "aggregation_threshold": threshold,
        "threshold_met": can_aggregate(connected_clients, threshold),
        "status": "ready" if can_aggregate(connected_clients, threshold) else "waiting"
    }

def log_threshold_status(round_number: int, connected_clients: int) -> str:
    """Log threshold status at each federated round."""
    threshold = get_threshold()
    met = can_aggregate(connected_clients, threshold)
    status = "MET" if met else "NOT MET"
    log_message = f"Round {round_number}: Threshold {status} ({connected_clients}/{threshold} clients)"
    print(log_message)
    return log_message
