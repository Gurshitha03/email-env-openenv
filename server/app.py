import sys
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from my_env import EmailEnv

env = EmailEnv(task="easy")


class Handler(BaseHTTPRequestHandler):

    def do_POST(self):
        global env

        if self.path == "/reset":
            obs = env.reset()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(obs).encode())

        elif self.path == "/step":
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))

            obs, reward, done, info = env.step(data)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info
            }).encode())

    def do_GET(self):
        if self.path == "/state":
            state = env.state()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OpenEnv running")


def main():
    PORT = 7860
    print(f"Server running on {PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
