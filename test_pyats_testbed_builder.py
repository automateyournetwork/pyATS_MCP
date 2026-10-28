"""
test_pyats_testbed_builder.py
=============================
Unit tests for pyats_testbed_builder.py. No network, no pyATS: HTTP is faked
and the module under test imports neither pyATS nor the MCP SDK.

Run with:
    pytest test_pyats_testbed_builder.py -v
"""

from __future__ import annotations

import json
import os

import pytest
import yaml

import pyats_testbed_builder as tbb


class FakeResp:
    def __init__(self, body, status=200, text=""):
        self._body, self.status_code, self.text = body, status, text

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)


class FakeSession:
    """Routes by URL substring; records calls."""

    def __init__(self, routes):
        self.routes, self.calls, self.verify = routes, [], True

    def _route(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for key, handler in self.routes.items():
            if key in url:
                return handler(kw) if callable(handler) else handler
        return FakeResp({}, 404)

    def get(self, url, **kw):
        return self._route("GET", url, **kw)

    def post(self, url, **kw):
        return self._route("POST", url, **kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("ITENTIAL_", "PLATFORM_", "NETBOX_", "CATALYST_", "PYATS_MCP_TESTBED")) or k in (
            "AUTH_METHOD", "CLIENT_ID", "CLIENT_SECRET",
        ):
            monkeypatch.delenv(k, raising=False)


# --- path / expression resolution ------------------------------------------
def test_get_path_nested_index_and_wildcard():
    rec = {"a": {"b": [{"c": 1}, {"c": 2}]}, "tags": [{"slug": "x"}, {"slug": "y"}]}
    assert tbb.get_path(rec, "a.b[1].c") == 2
    assert tbb.get_path(rec, "tags[*].slug") == ["x", "y"]
    assert tbb.get_path(rec, "a.zzz") is None
    assert tbb.get_path(rec, "a.b[9].c") is None


def test_resolve_template_literal_map_regex_default():
    rec = {"site": "ATL", "name": "r1", "v": "IOS-XE 17.3", "os": "ios-xe"}
    assert tbb.resolve(rec, "{site}-{name}") == "ATL-r1"
    assert tbb.resolve(rec, "literal:ssh") == "ssh"
    assert tbb.resolve(rec, {"path": "v", "regex": r"(\d+\.\d+)"}) == "17.3"
    assert tbb.resolve(rec, {"path": "os", "map": {"ios-xe": "iosxe"}}) == "iosxe"
    assert tbb.resolve(rec, {"path": "missing", "default": "d"}) == "d"


@pytest.mark.parametrize(
    "raw,expected,known",
    [
        ("IOS-XE", "iosxe", True),
        ("Cisco NX-OS", "nxos", True),
        ("frr", "linux", True),
        ("IOS-XR", "iosxr", True),
        ("Weird OS", "weird os", False),
    ],
)
def test_normalize_os(raw, expected, known):
    assert tbb.normalize_os(raw) == (expected, known)


def test_normalize_os_extra_map_wins():
    assert tbb.normalize_os("Weird OS", {"weird os": "linux"}) == ("linux", True)


# --- parsing ---------------------------------------------------------------
def test_parse_csv_json_yaml_xml():
    assert tbb.parse_payload("name,ip\nr1,1.1.1.1\n")[0]["ip"] == "1.1.1.1"
    assert tbb.parse_payload('{"devices": [{"name": "r1"}]}')["devices"][0]["name"] == "r1"
    assert tbb.parse_payload("- name: r1\n  ip: 1.1.1.1\n")[0]["name"] == "r1"
    x = tbb.parse_payload("<inv><dev><name>r1</name></dev><dev><name>r2</name></dev></inv>")
    assert [d["name"] for d in x["inv"]["dev"]] == ["r1", "r2"]


def test_parse_error_is_reported_not_raised_raw():
    with pytest.raises(tbb.TestbedBuildError, match="could not parse"):
        tbb.parse_payload("{bad json", "json")


def test_extract_records_heuristics():
    assert tbb.extract_records({"items": [{"name": "a"}]}, None) == [{"name": "a"}]
    assert tbb.extract_records({"results": [{"name": "a"}]}, "results") == [{"name": "a"}]
    tb = {"devices": {"r1": {"os": "iosxe"}}}
    assert tbb.extract_records(tb, None) == [{"name": "r1", "os": "iosxe"}]
    with pytest.raises(tbb.TestbedBuildError):
        tbb.extract_records({"x": 1}, "nope")


def test_detect_format():
    assert tbb.detect_format("<a/>") == "xml"
    assert tbb.detect_format('[{"a":1}]') == "json"
    assert tbb.detect_format("name,ip\nr1,1.1.1.1") == "csv"
    assert tbb.detect_format("a: 1") == "yaml"


# --- filters ---------------------------------------------------------------
def test_filters_include_exclude_and_list_fields():
    recs = [
        {"name": "a", "tags": ["leaf", "dc1"], "os": "x"},
        {"name": "b", "tags": ["spine"], "os": "x"},
        {"name": "c", "tags": ["leaf"], "os": "y"},
    ]
    out = tbb.apply_filters(
        recs, {"include": [{"field": "tags", "in": ["leaf"]}], "exclude": [{"field": "os", "equals": "y"}]}
    )
    assert [r["name"] for r in out] == ["a"]
    assert tbb.apply_filters(recs, {"include": [{"field": "name", "regex": "^[bc]$"}]})[0]["name"] == "b"
    with pytest.raises(tbb.TestbedBuildError):
        tbb.apply_filters(recs, {"include": [{"field": "name"}]})


# --- building --------------------------------------------------------------
NODES = [
    {
        "name": "dc1-leaf1",
        "_id": "1",
        "inventory_name": "inv",
        "tags": ["leaf"],
        "attributes": {
            "itential_host": "10.1.1.126/24",
            "itential_platform": "frr",
            "itential_port": 2213,
            "itential_password": "hunter2",
            "cluster_id": "c1",
        },
    },
    {"name": "no-ip", "tags": [], "attributes": {"itential_platform": "ios"}},
    {"name": "dc1-leaf1", "tags": [], "attributes": {"itential_host": "9.9.9.9", "itential_platform": "ios"}},
]


def _build(records=NODES, **kw):
    p = tbb.PRESETS["itential_inventory"]
    return tbb.build_testbed(records, p["mapping"], name="t", credentials=tbb.build_credentials(None, None), **kw)


def test_build_itential_preset_maps_skips_and_never_leaks_secrets():
    tb, rep = _build()
    dev = tb["devices"]["dc1-leaf1"]
    assert dev["os"] == "linux"
    assert dev["connections"]["cli"] == {"protocol": "ssh", "ip": "10.1.1.126", "port": 2213}
    assert dev["custom"]["cluster_id"] == "c1" and dev["custom"]["tags"] == ["leaf"]
    assert "hunter2" not in json.dumps(tb)
    assert rep["devices_built"] == 1 and rep["devices_skipped"] == 2
    reasons = {s["reason"] for s in rep["skipped_sample"]}
    assert reasons == {"no management address", "duplicate device name"}


def test_build_requires_os_or_default():
    recs = [{"name": "a", "tags": [], "attributes": {"itential_host": "1.1.1.1", "itential_platform": ""}}]
    with pytest.raises(tbb.TestbedBuildError, match="no os/platform"):
        _build(recs)
    tb, _ = _build(recs, default_os="iosxe")
    assert tb["devices"]["a"]["os"] == "iosxe"


def test_build_unknown_os_warns_and_os_map_fixes():
    recs = [{"name": "a", "tags": [], "attributes": {"itential_host": "1.1.1.1", "itential_platform": "Foo"}}]
    _, rep = _build(recs)
    assert any("not in the known alias table" in w for w in rep["warnings"])
    tb, rep = _build(recs, os_map={"Foo": "nxos"})
    assert tb["devices"]["a"]["os"] == "nxos" and not rep["warnings"]


def test_build_max_devices_caps():
    recs = [
        {"name": f"d{i}", "tags": [], "attributes": {"itential_host": f"10.0.0.{i}", "itential_platform": "ios"}}
        for i in range(10)
    ]
    tb, rep = _build(recs, max_devices=3)
    assert len(tb["devices"]) == 3 and any("max_devices" in w for w in rep["warnings"])


def test_netbox_and_catalyst_presets():
    nb = {
        "name": "sw1", "primary_ip4": {"address": "10.0.0.5/24"}, "platform": {"slug": "nx-os"},
        "role": {"slug": "access"}, "device_type": {"model": "9300"}, "site": {"slug": "atl"},
        "tags": [{"slug": "lab"}],
    }
    tb, _ = tbb.build_testbed([nb], tbb.PRESETS["netbox"]["mapping"], name="n", credentials={})
    d = tb["devices"]["sw1"]
    assert d["os"] == "nxos" and d["model"] == "9300" and d["custom"]["tags"] == ["lab"]

    cc = [
        {"hostname": "c1", "managementIpAddress": "1.1.1.1", "softwareType": "IOS-XE", "platformId": "C9300"},
        {"hostname": "wlc", "managementIpAddress": "2.2.2.2", "softwareType": "Cisco Controller"},
    ]
    p = tbb.PRESETS["catalyst_center"]
    kept = tbb.apply_filters(cc, p["filters"])
    tb, _ = tbb.build_testbed(kept, p["mapping"], name="c", credentials={})
    assert list(tb["devices"]) == ["c1"] and tb["devices"]["c1"]["os"] == "iosxe"


def test_csv_with_custom_mapping_end_to_end():
    recs = tbb.extract_records(tbb.parse_payload("host,addr,nos\nr1,10.0.0.1,ios-xe\nr2,10.0.0.2,nx-os\n"), None)
    tb, rep = tbb.build_testbed(
        recs, {"name": "host", "ip": "addr", "os": "nos"}, name="csv", credentials=tbb.build_credentials(None, None)
    )
    assert {n: d["os"] for n, d in tb["devices"].items()} == {"r1": "iosxe", "r2": "nxos"}
    assert rep["os_counts"] == {"iosxe": 1, "nxos": 1}


def test_mapping_requires_name_and_ip():
    with pytest.raises(tbb.TestbedBuildError, match="name' and 'ip"):
        tbb.build_testbed([{"a": 1}], {"name": "a"}, name="x", credentials={})


# --- credentials -----------------------------------------------------------
def test_credentials_are_env_refs_only():
    c = tbb.build_credentials({"username_env": "U", "password_env": "P", "enable_env": "E"}, None)
    assert c["default"] == {"username": "%ENV{U}", "password": "%ENV{P}"}
    assert c["enable"] == {"password": "%ENV{E}"}


@pytest.mark.parametrize("bad", ["hunter2!", "has space", "a-b", "%ENV{X}"])
def test_credentials_reject_inline_secret_shaped_values(bad):
    with pytest.raises(tbb.TestbedBuildError, match="environment variable NAME"):
        tbb.build_credentials({"password_env": bad}, None)


def test_credentials_from_static():
    static = {"testbed": {"credentials": {"default": {"username": "%ENV{X}", "password": "%ENV{Y}"}}}}
    assert tbb.build_credentials({"from": "static"}, static)["default"]["username"] == "%ENV{X}"
    with pytest.raises(tbb.TestbedBuildError):
        tbb.build_credentials({"from": "static"}, {"testbed": {}})


def test_redact():
    out = tbb.redact({"user": "u", "itential_password": "p", "nested": [{"api_token": "t", "ok": 1}]})
    assert out == {"user": "u", "itential_password": "<redacted>", "nested": [{"api_token": "<redacted>", "ok": 1}]}


def test_unresolved_env(monkeypatch):
    raw = {"testbed": {"credentials": {"default": {"username": "%ENV{A_SET}", "password": "%ENV{B_UNSET}"}}}}
    monkeypatch.setenv("A_SET", "1")
    monkeypatch.delenv("B_UNSET", raising=False)
    assert tbb.env_refs(raw) == ["A_SET", "B_UNSET"]
    assert tbb.unresolved_env(raw) == ["B_UNSET"]


# --- merge -----------------------------------------------------------------
def test_merge_base_wins_on_clash():
    base = {"testbed": {"name": "s"}, "devices": {"a": {"os": "ios"}}}
    extra = {"devices": {"a": {"os": "nxos"}, "b": {"os": "eos"}}}
    out, clashes = tbb.merge_raw(base, extra, "m")
    assert clashes == ["a"] and out["devices"]["a"]["os"] == "ios" and "b" in out["devices"]
    assert out["testbed"]["name"] == "m" and base["testbed"]["name"] == "s"


# --- registry --------------------------------------------------------------
def test_registry_lifecycle_and_materialize(tmp_path):
    reg = tbb.TestbedRegistry(tmp_path / "tb")
    raw = {"testbed": {"name": "x"}, "devices": {"a": {"os": "ios"}}}
    e = reg.put("x", raw, {})
    assert reg.active_entry() is None
    reg.activate("x")
    assert reg.active_entry() is e and reg.active_name == "x"
    path = reg.materialize(e)
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    assert yaml.safe_load(open(path))["devices"]["a"]["os"] == "ios"
    assert reg.materialize(e) == path
    reg.put("x", raw, {})  # replace removes the old file
    assert not os.path.exists(path)
    reg.delete("x")
    assert reg.active_entry() is None
    with pytest.raises(tbb.TestbedBuildError):
        reg.get("x")
    with pytest.raises(tbb.TestbedBuildError):
        reg.put("../evil", raw, {})
    with pytest.raises(tbb.TestbedBuildError):
        reg.activate("nope")


def test_registry_capacity(tmp_path):
    reg = tbb.TestbedRegistry(tmp_path)
    for i in range(tbb.MAX_REGISTRY_ENTRIES):
        reg.put(f"t{i}", {"devices": {}}, {})
    with pytest.raises(tbb.TestbedBuildError, match="registry full"):
        reg.put("overflow", {"devices": {}}, {})
    reg.put("t0", {"devices": {}}, {})  # replacing is still fine


def test_tcp_probe_localhost():
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    raw = {"devices": {"up": {"connections": {"defaults": {"via": "cli"}, "cli": {"ip": "127.0.0.1", "port": port}}},
                       "down": {"connections": {"cli": {"ip": "127.0.0.1", "port": 1}}}}}
    try:
        res = tbb.tcp_probe(raw, timeout=1)
    finally:
        srv.close()
    assert res["checked"] == 2 and res["reachable"] == 1 and "down" in res["unreachable"]


# --- Itential --------------------------------------------------------------
def _itential_env(monkeypatch, prefix=""):
    if prefix:
        monkeypatch.setenv(f"ITENTIAL_{prefix}_URL", "https://dev.example.io/")
        monkeypatch.setenv(f"ITENTIAL_{prefix}_AUTH_METHOD", "oauth")
        monkeypatch.setenv(f"ITENTIAL_{prefix}_CLIENT_ID", "id")
        monkeypatch.setenv(f"ITENTIAL_{prefix}_CLIENT_SECRET", "sec")
    else:
        monkeypatch.setenv("PLATFORM_URL", "https://dev.example.io/")
        monkeypatch.setenv("AUTH_METHOD", "oauth")
        monkeypatch.setenv("CLIENT_ID", "id")
        monkeypatch.setenv("CLIENT_SECRET", "sec")


def _inv_session(pages):
    def nodes(kw):
        page = kw["params"]["page"]
        return FakeResp({"status": "Success", "result": {"data": pages[page - 1], "totalPages": len(pages)}})

    return FakeSession(
        {
            "/oauth/token": FakeResp({"access_token": "TOK"}),
            "/nodes": nodes,
            "/inventories?": FakeResp({}),
        }
    )


def test_itential_paginates_oauth_and_tags_inventory(monkeypatch):
    _itential_env(monkeypatch)
    sess = _inv_session([[{"name": "a", "attributes": {}}], [{"name": "b", "attributes": {}}]])
    recs = tbb.fetch_itential(["inv one"], None, 100, client=tbb.ItentialClient(session=sess))
    assert [r["name"] for r in recs] == ["a", "b"]
    assert all(r["inventory_name"] == "inv one" for r in recs)
    node_calls = [c for c in sess.calls if "/nodes" in c[1]]
    assert "inv%20one" in node_calls[0][1]  # inventory name is URL-quoted
    assert node_calls[0][2]["headers"] == {"Authorization": "Bearer TOK"}
    assert sess.calls[0][2]["data"]["grant_type"] == "client_credentials"


def test_itential_max_nodes_caps(monkeypatch):
    _itential_env(monkeypatch)
    sess = _inv_session([[{"name": str(i), "attributes": {}} for i in range(5)], [{"name": "z", "attributes": {}}]])
    recs = tbb.fetch_itential(["i"], None, 3, client=tbb.ItentialClient(session=sess))
    assert len(recs) == 3


def test_itential_profile_uses_prefixed_env(monkeypatch):
    _itential_env(monkeypatch, "DEVSTACK")
    cl = tbb.ItentialClient("devstack", session=FakeSession({}))
    assert cl.cfg["URL"] == "https://dev.example.io"


def test_itential_unconfigured_is_a_clear_error():
    with pytest.raises(tbb.TestbedBuildError, match="not configured"):
        tbb.ItentialClient(None)
    with pytest.raises(tbb.TestbedBuildError, match="letters, digits"):
        tbb.itential_config("bad profile!")


def test_itential_unknown_inventory_lists_available(monkeypatch):
    _itential_env(monkeypatch)
    sess = FakeSession(
        {
            "/oauth/token": FakeResp({"access_token": "T"}),
            "/nodes": FakeResp({}, 404),
            "/inventories": FakeResp({"result": {"data": [{"name": "real"}], "totalPages": 1}}),
        }
    )
    with pytest.raises(tbb.TestbedBuildError, match=r"available: \['real'\]"):
        tbb.fetch_itential(["nope"], None, 10, client=tbb.ItentialClient(session=sess))


def test_itential_reauths_on_401(monkeypatch):
    _itential_env(monkeypatch)
    state = {"n": 0}

    def nodes(kw):
        state["n"] += 1
        if state["n"] == 1:
            return FakeResp({}, 401)
        return FakeResp({"result": {"data": [{"name": "a", "attributes": {}}], "totalPages": 1}})

    sess = FakeSession({"/oauth/token": FakeResp({"access_token": "T"}), "/nodes": nodes})
    recs = tbb.fetch_itential(["i"], None, 10, client=tbb.ItentialClient(session=sess))
    assert len(recs) == 1 and sum("/oauth/token" in c[1] for c in sess.calls) == 2


# --- generic REST ----------------------------------------------------------
def test_rest_host_allowlist(monkeypatch):
    with pytest.raises(tbb.TestbedBuildError, match="not an allowed source"):
        tbb.fetch_rest("https://evil.example.com/x", session=FakeSession({}))
    with pytest.raises(tbb.TestbedBuildError, match="absolute http"):
        tbb.fetch_rest("file:///etc/passwd", session=FakeSession({}))
    monkeypatch.setenv("PYATS_MCP_TESTBED_SOURCE_HOSTS", "ipam.example.com")
    sess = FakeSession({"ipam.example.com": FakeResp({"items": [{"name": "a"}]})})
    assert tbb.fetch_rest("https://ipam.example.com/d", records_path="items", session=sess) == [{"name": "a"}]


def test_rest_header_secrets_come_from_env_names(monkeypatch):
    monkeypatch.setenv("PYATS_MCP_TESTBED_SOURCE_HOSTS", "ipam.example.com")
    monkeypatch.setenv("IPAM_AUTH", "Token abc")
    sess = FakeSession({"ipam.example.com": FakeResp([{"name": "a"}])})
    tbb.fetch_rest("https://ipam.example.com/d", headers_env={"Authorization": "IPAM_AUTH"}, session=sess)
    assert sess.calls[0][2]["headers"] == {"Authorization": "Token abc"}
    with pytest.raises(tbb.TestbedBuildError, match="not set on the server"):
        tbb.fetch_rest("https://ipam.example.com/d", headers_env={"Authorization": "NOPE_UNSET"}, session=sess)
    with pytest.raises(tbb.TestbedBuildError, match="environment variable NAME"):
        tbb.fetch_rest("https://ipam.example.com/d", headers_env={"Authorization": "Token abc"}, session=sess)


def test_rest_next_url_pagination_refuses_cross_host(monkeypatch):
    monkeypatch.setenv("PYATS_MCP_TESTBED_SOURCE_HOSTS", "ipam.example.com")
    pages = {
        "/p1": FakeResp({"results": [{"n": 1}], "next": "https://ipam.example.com/p2"}),
        "/p2": FakeResp({"results": [{"n": 2}], "next": "https://attacker.example.net/p3"}),
    }
    out = tbb.fetch_rest(
        "https://ipam.example.com/p1",
        records_path="results",
        pagination={"type": "next_url", "path": "next"},
        session=FakeSession(pages),
    )
    assert [r["n"] for r in out] == [1, 2]


def test_rest_offset_pagination(monkeypatch):
    monkeypatch.setenv("PYATS_MCP_TESTBED_SOURCE_HOSTS", "ipam.example.com")

    def handler(kw):
        off = kw["params"]["offset"]
        return FakeResp({"data": [{"n": i} for i in range(off, min(off + 2, 5))]})

    out = tbb.fetch_rest(
        "https://ipam.example.com/d",
        records_path="data",
        pagination={"type": "offset", "size": 2},
        session=FakeSession({"ipam.example.com": handler}),
    )
    assert [r["n"] for r in out] == [0, 1, 2, 3, 4]


# --- files -----------------------------------------------------------------
def test_read_source_file_confined_to_allowed_dirs(tmp_path):
    good = tmp_path / "ok" / "inv.csv"
    good.parent.mkdir()
    good.write_text("name,ip\na,1.1.1.1\n")
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    dirs = [good.parent.resolve()]
    assert "name,ip" in tbb.read_source_file(str(good), dirs)
    with pytest.raises(tbb.TestbedBuildError, match="outside the allowed"):
        tbb.read_source_file(str(outside), dirs)
    with pytest.raises(tbb.TestbedBuildError, match="outside the allowed"):
        tbb.read_source_file(str(good.parent / ".." / "secret.txt"), dirs)
