import os
import sys
import json
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer

# Allow importing my_env.py from the project root
sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

from my_env import EmailEnv


# Path to server folder
BASE_DIR = os.path.dirname(__file__)

# Default environment
env = EmailEnv(task="easy")


class Handler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json"
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.end_headers()

        self.wfile.write(response)

    # --------------------------------
    # GET REQUESTS
    # --------------------------------

    def do_GET(self):

        # Remove query parameters from URL
        # Example:
        # /?logs=container&__theme=system
        # becomes /
        path = urlparse(self.path).path

        # Serve the UI
        if path == "/" or path == "/index.html":

            try:

                with open(
                    os.path.join(BASE_DIR, "index.html"),
                    "rb"
                ) as file:

                    html = file.read()

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(html))
                )

                self.end_headers()

                self.wfile.write(html)

            except Exception as e:

                self.send_response(500)

                self.send_header(
                    "Content-Type",
                    "text/plain"
                )

                self.end_headers()

                self.wfile.write(
                    f"Error loading UI: {e}".encode(
                        "utf-8"
                    )
                )

            return

        # Return environment state
        if path == "/state":

            self.send_json(
                env.state()
            )

            return

        # Unknown GET endpoint
        self.send_json(
            {
                "error": "Endpoint not found"
            },
            404
        )


    # --------------------------------
    # POST REQUESTS
    # --------------------------------

    def do_POST(self):

        global env

        # Read request body
        length = int(
            self.headers.get(
                "Content-Length",
                0
            )
        )

        body = self.rfile.read(length)

        # Parse JSON
        try:

            data = (
                json.loads(body)
                if body
                else {}
            )

        except json.JSONDecodeError:

            self.send_json(
                {
                    "error": "Invalid JSON"
                },
                400
            )

            return


        # -----------------------------
        # RESET / START TASK
        # -----------------------------

        if self.path == "/reset":

            task = data.get(
                "task",
                "easy"
            )

            # Validate task
            if task not in [
                "easy",
                "medium",
                "hard"
            ]:

                self.send_json(
                    {
                        "error": "Invalid task"
                    },
                    400
                )

                return

            # Create new environment
            env = EmailEnv(
                task=task
            )

            # Reset environment
            observation = env.reset()

            self.send_json(
                observation
            )

            return


        # -----------------------------
        # STEP / CLASSIFICATION
        # -----------------------------

        if self.path == "/step":

            try:

                category = data.get(
                    "category",
                    ""
                )

                response = data.get(
                    "response",
                    category
                )

                action = {
                    "category": category,
                    "response": response
                }

                observation, reward, done, info = env.step(
                    action
                )

                self.send_json(
                    {
                        "observation": observation,
                        "reward": reward,
                        "done": done,
                        "info": info
                    }
                )

            except Exception as e:

                self.send_json(
                    {
                        "error": str(e)
                    },
                    500
                )

            return


        # Unknown POST endpoint
        self.send_json(
            {
                "error": "Endpoint not found"
            },
            404
        )


# --------------------------------
# SERVER
# --------------------------------

def main():

    PORT = 7860

    print(
        f"Server running on {PORT}"
    )

    server = HTTPServer(
        ("0.0.0.0", PORT),
        Handler
    )

    server.serve_forever()


# --------------------------------
# START APPLICATION
# --------------------------------

if __name__ == "__main__":

    main()
