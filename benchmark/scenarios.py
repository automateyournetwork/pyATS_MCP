"""
benchmark/scenarios.py
=======================
Shared scenario definitions for the pre/post transport benchmark.

Each scenario is (name, tool_name, kwargs). Kept in one place so the
STDIO-legacy runner and the Streamable-HTTP runner exercise identically
shaped calls.
"""

DEVICES = ["R1", "R2", "SW1", "SW2"]

SCENARIOS = [
    ("single_show_command", "pyats_run_show_command",
     {"device_name": "R1", "command": "show version"}),
    ("multi_show_thread_pool", "pyats_run_show_command_multi",
     {"device_names": DEVICES, "command": "show ip interface brief"}),
    ("multi_show_pcall", "pyats_pcall_show_command",
     {"device_names": DEVICES, "command": "show ip interface brief"}),
    ("device_health", "pyats_device_health",
     {"device_name": "R1"}),
]

WARMUP_ITERATIONS = 2
MEASURED_ITERATIONS = 8
