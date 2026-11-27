"""Serve the web example to a local browser and print what the page reports.

    uv run python serve.py                       # torchnative from PyPI
    uv run python serve.py --wheel dist/torchnative-*-pyemscripten_2026_0_wasm32.whl
    uv run python serve.py --no-browser          # print the URL, open it yourself

It serves this folder over loopback (the page fetches `basics.py` and `reference.json`,
and a page opened as a file cannot), serves `--wheel` under `wheels/`, opens the page,
waits for the page to POST its result, prints the page's lines, and exits with the
page's status (0 when every case agrees with upstream torch).
"""

import argparse
import http.server
import json
import queue
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVED = ("index.html", "basics.py", "reference.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, help="a torchnative pyemscripten wheel to install instead of PyPI's")
    parser.add_argument("--port", type=int, default=8737)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for the page")
    args = parser.parse_args(argv)
    if args.wheel and not args.wheel.is_file():
        parser.error(f"no wheel at {args.wheel}")
    results = queue.Queue()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path.lstrip("/") or "index.html"
            if path in SERVED:
                body = (HERE / path).read_bytes()
            elif args.wheel and path == f"wheels/{args.wheel.name}":
                body = args.wheel.read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            kind = {".html": "text/html", ".json": "application/json"}.get(Path(path).suffix, "application/octet-stream")
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/result":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(204)
            self.end_headers()
            results.put(json.loads(body))

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    if args.wheel:
        url += "?wheel=" + urllib.parse.quote(f"wheels/{args.wheel.name}")
    print(f"serving {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        result = results.get(timeout=args.timeout)
    except queue.Empty:
        print(f"no result from the page within {args.timeout:.0f} s", file=sys.stderr)
        return 2
    finally:
        server.shutdown()
    for line in result["lines"]:
        print(line)
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
