#!/usr/bin/env python3
"""Static file server with on-the-fly gzip + sensible caching, for the dashboard.

    python3 scripts/serve.py [port]      (default 8765, binds 0.0.0.0)

gzip shrinks the JSON/geojson ~5x and the binaries ~10-25%, cutting load time
a lot over the SSH tunnel. Threaded so parallel asset fetches don't serialize.
"""
import gzip, io, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPRESSIBLE = (".html", ".js", ".css", ".json", ".geojson", ".bin", ".svg", ".csv")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        # let the browser cache immutable data assets; keep html fresh
        if self.path.startswith("/dashboard/data/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        # We must rewrite Content-Length when gzipping, so replicate minimally.
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        if not (accepts_gzip and self.path.lower().endswith(COMPRESSIBLE)):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        data = f.read(); f.close()
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
            gz.write(data)
        gzdata = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(gzdata)))
        if self.path.startswith("/dashboard/data/"):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return io.BytesIO(gzdata)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving {ROOT} on 0.0.0.0:{port} (gzip enabled)")
    httpd.serve_forever()
