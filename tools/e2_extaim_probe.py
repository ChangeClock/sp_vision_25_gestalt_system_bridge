#!/usr/bin/env python3
"""E2/E3 probe: does RBExtAim really steer an AI-controlled turret, and does the
fire pulse shoot through the normal gates?

Protocol (on the live blue sentry, soft-claimed):
  - keeper thread: BatchSet AITargetMode=90 (external mode) + AIMoveMode=2 (hold
    position) at 5Hz so the strategy layer can't reclaim aiming.
  - E2 step sweep: command world yaws [+90, 180, -90, 0] (pitch 0), hold each 6s
    at 10Hz RBExtAim, sample telemetry TurretYaw at 4Hz; report settle error and
    hold jitter per step.
  - E3 fire: aim steady, then RBExtAim with fire=1 at 10Hz for 6s; report
    BulletFiredTotal delta plus ammo/heat movement (gates respected = bullets
    stop when allowance/heat says so, not when we say so).

Usage: python tools/e2_extaim_probe.py <wsPort>
"""

import json
import sys
import threading
import time

from websockets.sync.client import connect as ws_connect

A = {
    "pid": 10000035, "team": 10000036, "cls": 60000002,
    "tyaw": 10000111, "tpitch": 10000112,
    "bullets": 63000002, "allow": 10000033, "real": 10000031,
    "heat": 10000011, "defeated": 50000007,
    "target_mode": 50000090, "move_mode": 50000089, "fire_rate": 50000091,
    "match": 80000005, "ptr": 1000001,
}


class Link:
    def __init__(self, port):
        self.ws = ws_connect(f"ws://127.0.0.1:{port}/", max_size=None)
        self.maps = {}
        self.watched = set()
        self.next_id = 970001
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()
        self.watch(list(range(1, 257)))

    def _reader(self):
        for raw in self.ws:
            try:
                p = json.loads(raw)
            except Exception:
                continue
            if p.get("type") == 0 and p.get("method") == "watchAttributeMaps.result":
                found = []
                with self.lock:
                    for r in p.get("params", {}).get("watch_attribute_maps_results", []):
                        m = self.maps.setdefault(r["attribute_map_id"], {})
                        for k, v in (r.get("attributes") or {}).items():
                            m[int(k)] = v
                            if int(k) == A["ptr"] and isinstance(v, (int, float)) and v > 0:
                                found.append(int(v))
                if found:
                    self.watch(found)

    def watch(self, ids):
        fresh = [i for i in ids if i not in self.watched]
        if not fresh:
            return
        self.watched.update(fresh)
        self._send("attribute.watchAttributeMaps",
                   {"attribute_map_ids": fresh, "watch_type": 2})

    def _send(self, method, params):
        with self.lock:
            self.next_id += 1
            msg = {"type": 0, "id": self.next_id, "method": method, "params": params}
        self.ws.send(json.dumps(msg))

    def exec(self, cmd):
        self._send("console.exec", {"command": cmd})

    def attr(self, mid, aid):
        with self.lock:
            return self.maps.get(mid, {}).get(aid)


def norm180(d):
    return (d + 180.0) % 360.0 - 180.0


def main():
    port = int(sys.argv[1])
    link = Link(port)
    print(f"[e2] connected :{port}")

    sentry = pid = None
    for _ in range(240):
        with link.lock:
            for mid, m in link.maps.items():
                if m.get(A["cls"]) == 1004 and m.get(A["team"]) == 1:
                    sentry, pid = mid, m.get(A["pid"])
            running = any(m.get(A["match"]) == 1 for m in link.maps.values())
        if sentry and running:
            break
        time.sleep(0.5)
    if not sentry:
        raise SystemExit("[e2] blue sentry not found / match not running")
    print(f"[e2] blue sentry pid={pid} map={sentry}")

    def keeper():
        while True:
            link.exec(f"BatchSet {A['target_mode']} 90 blue sentry")
            link.exec(f"BatchSet {A['move_mode']} 2 blue sentry")
            time.sleep(0.2)

    threading.Thread(target=keeper, daemon=True).start()
    time.sleep(2)

    results = {"steps": [], "fire": None}
    for step_yaw in (90.0, 180.0, -90.0, 0.0):
        t_end = time.time() + 6.0
        samples = []
        while time.time() < t_end:
            link.exec(f"UEExec RBExtAim {pid} {step_yaw} 0")
            y = link.attr(sentry, A["tyaw"])
            if y is not None:
                samples.append(norm180(y - step_yaw))
            time.sleep(0.1)
        tail = samples[-8:]
        step = {
            "cmd_yaw": step_yaw,
            "settle_err_deg": round(sum(abs(e) for e in tail) / len(tail), 2) if tail else None,
            "max_tail_err": round(max(abs(e) for e in tail), 2) if tail else None,
            "samples": len(samples),
        }
        results["steps"].append(step)
        print(f"[e2] step {step}")

    b0 = link.attr(sentry, A["bullets"]) or 0
    a0 = link.attr(sentry, A["allow"]) or 0
    h_max = 0
    t_end = time.time() + 6.0
    while time.time() < t_end:
        link.exec(f"UEExec RBExtAim {pid} 0 0 1")
        h = link.attr(sentry, A["heat"]) or 0
        h_max = max(h_max, h)
        time.sleep(0.1)
    time.sleep(1)
    b1 = link.attr(sentry, A["bullets"]) or 0
    a1 = link.attr(sentry, A["allow"]) or 0
    results["fire"] = {
        "bullets_delta": b1 - b0, "allowance_delta": a1 - a0,
        "heat_peak": h_max, "defeated": link.attr(sentry, A["defeated"]),
    }
    print(f"[e2] fire {results['fire']}")

    # Hand aiming back to the built-in AI.
    link.exec(f"BatchSet {A['target_mode']} 1 blue sentry")
    link.exec(f"BatchSet {A['move_mode']} 1 blue sentry")

    ok_steps = [s for s in results["steps"] if s["settle_err_deg"] is not None and s["settle_err_deg"] < 3.0]
    verdict = "PASS" if len(ok_steps) >= 3 and results["fire"]["bullets_delta"] > 0 else "PARTIAL"
    print(json.dumps({"verdict": verdict, **results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
