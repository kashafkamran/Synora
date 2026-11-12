# SBFLT-19: Tests for aggregation participation threshold

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.server.services.updates import can_aggregate, get_threshold, set_threshold
from backend.server.services.overview import get_server_overview, log_threshold_status

# --- Tests for updates.py ---

def test_threshold_met():
    assert can_aggregate(3, threshold=2) == True

def test_threshold_exactly_met():
    assert can_aggregate(2, threshold=2) == True

def test_threshold_not_met():
    assert can_aggregate(1, threshold=2) == False

def test_get_threshold_default():
    result = get_threshold()
    assert isinstance(result, int)
    assert result >= 1

def test_set_threshold_valid():
    result = set_threshold(3)
    assert result["success"] == True
    assert result["threshold"] == 3

def test_set_threshold_invalid():
    result = set_threshold(0)
    assert result["success"] == False

# --- Tests for overview.py ---

def test_overview_threshold_met():
    overview = get_server_overview(5)
    assert overview["threshold_met"] == True
    assert overview["status"] == "ready"

def test_overview_threshold_not_met():
    overview = get_server_overview(0)
    assert overview["threshold_met"] == False
    assert overview["status"] == "waiting"

def test_log_threshold_status():
    log = log_threshold_status(1, 3)
    assert "Round 1" in log
