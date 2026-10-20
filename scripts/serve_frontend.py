"""Serve the research console on the local network."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from factor_lab.service import ResearchService


FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend"
PROJECT_ROOT = FRONTEND_ROOT.parent
RESEARCH_SERVICE = ResearchService(PROJECT_ROOT)


def json_response(payload: object, status: int = 200) -> tuple[int, bytes]:
    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, "item"):
            return json_safe(value.item())
        return value

    return status, (json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


class ResearchConsoleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/healthz":
            body = json.dumps({"status": "ok", "service": "crypto-factor-lab"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            self.send_json(RESEARCH_SERVICE.status())
            return
        if path == "/api/factors":
            self.send_json({"factors": RESEARCH_SERVICE.factors()})
            return
        if path == "/api/factor-library":
            self.send_json({"factors": RESEARCH_SERVICE.factor_library()})
            return
        if path == "/api/observations":
            self.send_json({"tracks": RESEARCH_SERVICE.observation_tracks()})
            return
        if path == "/api/experiments":
            state = RESEARCH_SERVICE.status()
            self.send_json({"historical_reports": state["latest_report"], "tasks": state["tasks"]})
            return
        if path == "/api/mining":
            self.send_json({"runs": RESEARCH_SERVICE.mining_runs()})
            return
        if path.startswith("/api/mining/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                self.send_json(RESEARCH_SERVICE.mining_run(parts[2]))
                return
            if len(parts) == 5 and parts[3] == "candidates":
                self.send_json(RESEARCH_SERVICE.mining_candidate(parts[2], parts[4]))
                return
            self.send_json({"error": "not found"}, status=404)
            return
        if path == "/api/tasks":
            self.send_json({"tasks": RESEARCH_SERVICE.task_history()})
            return
        if path.startswith("/api/tasks/"):
            task_id = path.rsplit("/", maxsplit=1)[-1]
            task = RESEARCH_SERVICE.task(task_id)
            self.send_json({"task_id": task.task_id, "source": "web_task", "status": task.status, "submitted_at": task.submitted_at, "started_at": task.started_at, "finished_at": task.finished_at, "payload": task.payload, "result": task.result, "error": task.error})
            return
        if path == "/api/chart":
            self.send_json(RESEARCH_SERVICE.chart())
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        promotion_prefix = "/api/mining/"
        is_promotion = path.startswith(promotion_prefix) and path.endswith("/promote")
        is_registration = path.startswith(promotion_prefix) and path.endswith("/register")
        is_observation = path.startswith(promotion_prefix) and path.endswith("/observe")
        is_observation_update = path.startswith("/api/observations/") and path.endswith("/update")
        if path not in ("/api/experiments", "/api/mining", "/api/data/update") and not is_promotion and not is_registration and not is_observation and not is_observation_update:
            self.send_json({"error": "not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if is_observation_update:
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    raise ValueError("invalid observation update path")
                self.send_json(RESEARCH_SERVICE.update_observation_track(parts[2]))
                return
            if is_promotion or is_registration or is_observation:
                parts = path.strip("/").split("/")
                expected_action = "register" if is_registration else "observe" if is_observation else "promote"
                if len(parts) != 6 or parts[0:2] != ["api", "mining"] or parts[3] != "candidates" or parts[5] != expected_action:
                    raise ValueError(f"invalid {expected_action} path")
                if is_registration:
                    result = RESEARCH_SERVICE.register_candidate(parts[2], parts[4], payload)
                    self.send_json(result, status=201)
                    return
                if is_observation:
                    self.send_json(
                        RESEARCH_SERVICE.create_observation_track(parts[2], parts[4], payload),
                        status=201,
                    )
                    return
                task = RESEARCH_SERVICE.submit_promotion(parts[2], parts[4], payload)
            else:
                task = RESEARCH_SERVICE.submit_mining(payload) if path == "/api/mining" else RESEARCH_SERVICE.submit_data_update() if path == "/api/data/update" else RESEARCH_SERVICE.submit_experiment(payload)
            self.send_json({"task_id": task.task_id, "status": task.status}, status=202)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)

    def send_json(self, payload: object, status: int = 200) -> None:
        status, body = json_response(payload, status)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the Crypto Factor Lab frontend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    arguments = parser.parse_args()

    server = ThreadingHTTPServer((arguments.host, arguments.port), ResearchConsoleHandler)
    print(f"Crypto Factor Lab: http://{arguments.host}:{arguments.port}", flush=True)
    print(f"Frontend root: {FRONTEND_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
