"""A tiny web service. It keeps running until stopped (Ctrl+C)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ok", "version": "1.4.2"} if self.path == "/health" else {"error": "not found"})
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


print("Serving on http://localhost:8000", flush=True)
HTTPServer(("localhost", 8000), Handler).serve_forever()
