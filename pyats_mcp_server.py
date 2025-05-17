# pyats_fastmcp_server.py

import os
import sys
import json
import logging
import asyncio
import textwrap
from functools import partial
from dotenv import load_dotenv
from pyats.topology import loader
from pydantic import BaseModel, Field
from typing import Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Load environment and testbed
load_dotenv()
TESTBED_PATH = os.getenv("PYATS_TESTBED_PATH")

if not TESTBED_PATH or not os.path.exists(TESTBED_PATH):
    raise SystemExit(f"❌ PYATS_TESTBED_PATH not set or file missing: {TESTBED_PATH}")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PyatsFastMCP")
logger.info(f"✅ Using testbed file: {TESTBED_PATH}")

# --------------------------- Input Schemas ---------------------------

class DeviceCommandInput(BaseModel):
    device_name: str = Field(..., description="Device name from testbed")
    command: str = Field(..., description="CLI command to run")

class ConfigInput(BaseModel):
    device_name: str = Field(...)
    config_commands: str = Field(...)

class DeviceOnlyInput(BaseModel):
    device_name: str = Field(...)

class LinuxCommandInput(BaseModel):
    device_name: str = Field(...)
    command: str = Field(...)

# --------------------------- Helper Functions ---------------------------

def _get_device(device_name: str):
    testbed = loader.load(TESTBED_PATH)
    device = testbed.devices.get(device_name)
    if not device:
        raise ToolError(f"Device '{device_name}' not found in testbed")
    if not device.is_connected():
        logger.info(f"Connecting to {device_name}...")
        device.connect()
    return device

def _disconnect_device(device):
    if device and device.is_connected():
        device.disconnect()

# --------------------------- Tool Functions ---------------------------

async def pyats_run_show_command(params: DeviceCommandInput) -> dict:
    device = _get_device(params.device_name)
    try:
        if not params.command.lower().strip().startswith("show"):
            raise ToolError("Only 'show' commands are allowed")
        try:
            parsed = device.parse(params.command)
            return {"status": "parsed", "output": parsed}
        except Exception:
            raw = device.execute(params.command)
            return {"status": "raw", "output": raw}
    finally:
        _disconnect_device(device)

async def pyats_configure_device(params: ConfigInput) -> dict:
    device = _get_device(params.device_name)
    try:
        config = textwrap.dedent(params.config_commands.strip())
        if not config:
            raise ToolError("Empty config")
        output = device.configure(config)
        return {"status": "configured", "output": output}
    finally:
        _disconnect_device(device)

async def pyats_show_running_config(params: DeviceOnlyInput) -> dict:
    device = _get_device(params.device_name)
    try:
        output = device.execute("show running-config")
        return {"status": "raw", "output": output}
    finally:
        _disconnect_device(device)

async def pyats_show_logging(params: DeviceOnlyInput) -> dict:
    device = _get_device(params.device_name)
    try:
        output = device.execute("show logging last 250")
        return {"status": "raw", "output": output}
    finally:
        _disconnect_device(device)

async def pyats_ping_from_network_device(params: DeviceCommandInput) -> dict:
    device = _get_device(params.device_name)
    try:
        if not params.command.strip().lower().startswith("ping"):
            raise ToolError("Only 'ping' commands are supported")
        try:
            parsed = device.parse(params.command)
            return {"status": "parsed", "output": parsed}
        except Exception:
            raw = device.execute(params.command)
            return {"status": "raw", "output": raw}
    finally:
        _disconnect_device(device)

async def pyats_run_linux_command(params: LinuxCommandInput) -> dict:
    device = _get_device(params.device_name)
    try:
        try:
            output = device.parse(params.command)
        except Exception:
            output = device.execute(params.command)
        return {"status": "completed", "output": output}
    finally:
        _disconnect_device(device)

# --------------------------- FastMCP Server ---------------------------

mcp = FastMCP(name="pyATS MCP Server", instructions="Use these tools to manage Cisco IOS and Linux devices via pyATS.")

@mcp.tool(name="pyats_run_show_command", description="Executes a Cisco IOS/NX-OS 'show' command. Returns parsed or raw output.")
async def pyats_run_show_command(params: DeviceCommandInput) -> dict:
    device = _get_device(params.device_name)
    try:
        if not params.command.lower().strip().startswith("show"):
            raise ToolError("Only 'show' commands are allowed")
        try:
            parsed = device.parse(params.command)
            return {"status": "parsed", "output": parsed}
        except Exception:
            raw = device.execute(params.command)
            return {"status": "raw", "output": raw}
    finally:
        _disconnect_device(device)

@mcp.tool(name="pyats_configure_device", description="Applies configuration commands to a Cisco IOS/NX-OS device.")
async def pyats_configure_device(params: ConfigInput) -> dict:
    device = _get_device(params.device_name)
    try:
        config = textwrap.dedent(params.config_commands.strip())
        if not config:
            raise ToolError("Empty config")
        output = device.configure(config)
        return {"status": "configured", "output": output}
    finally:
        _disconnect_device(device)

@mcp.tool(name="pyats_show_running_config", description="Retrieves the running configuration from a Cisco IOS/NX-OS device.")
async def pyats_show_running_config(params: DeviceOnlyInput) -> dict:
    device = _get_device(params.device_name)
    try:
        output = device.execute("show running-config")
        return {"status": "raw", "output": output}
    finally:
        _disconnect_device(device)

@mcp.tool(name="pyats_show_logging", description="Retrieves recent system logs from a Cisco IOS/NX-OS device.")
async def pyats_show_logging(params: DeviceOnlyInput) -> dict:
    device = _get_device(params.device_name)
    try:
        output = device.execute("show logging last 250")
        return {"status": "raw", "output": output}
    finally:
        _disconnect_device(device)

@mcp.tool(name="pyats_ping_from_network_device", description="Executes a 'ping' command on a Cisco IOS/NX-OS device.")
async def pyats_ping_from_network_device(params: DeviceCommandInput) -> dict:
    device = _get_device(params.device_name)
    try:
        if not params.command.strip().lower().startswith("ping"):
            raise ToolError("Only 'ping' commands are supported")
        try:
            parsed = device.parse(params.command)
            return {"status": "parsed", "output": parsed}
        except Exception:
            raw = device.execute(params.command)
            return {"status": "raw", "output": raw}
    finally:
        _disconnect_device(device)

@mcp.tool(name="pyats_run_linux_command", description="Executes a Linux command on a testbed-defined Linux device.")
async def pyats_run_linux_command(params: LinuxCommandInput) -> dict:
    device = _get_device(params.device_name)
    try:
        try:
            output = device.parse(params.command)
        except Exception:
            output = device.execute(params.command)
        return {"status": "completed", "output": output}
    finally:
        _disconnect_device(device)

if __name__ == "__main__":
    asyncio.run(mcp.run_async())
