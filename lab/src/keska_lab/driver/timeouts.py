"""Fail-fast timeout constants (single source of truth).

These are intentionally aggressive; the driver should never sit silently for minutes.
"""

POLL_INTERVAL_S = 0.5
POLL_STALL_S = 10.0

# CRI subprocess hard caps (seconds)
CRI_RUNP_S = 45.0
CRI_CREATE_S = 30.0
CRI_START_S = 30.0
CRI_STOP_S = 45.0
CRI_EXEC_S = 20.0
CRI_INSPECT_S = 10.0
CRI_LOGS_S = 10.0

# TSOT gRPC deadlines (seconds)
TSOT_CREATE_POD_S = 15.0
TSOT_RPC_DEFAULT_S = 10.0

# Readiness waits (Layer 1 caps)
SANDBOX_IP_READY_S = 20.0
CONTAINER_EXEC_READY_S = 20.0
IPERF_SERVER_READY_S = 30.0
IPERF_CLIENT_S = 45.0

# Budgets (outer layers)
TEARDOWN_BUDGET_S = 60.0
OP_BUDGET_INET_CONNECT_S = 90.0
OP_BUDGET_INET_DOWNLOAD_S = 120.0
OP_BUDGET_IPERF_PAIR_S = 300.0
SSH_SLACK_S = 30.0

