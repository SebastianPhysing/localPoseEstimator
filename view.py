# for visualization with browser running three.js

# This script is partly written with the help of AI.

import os
import argparse
import functools
import http.server
import webbrowser


VIEWER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    '''
    Serves the files of the viewer folder. /data.json is poses3d.json of the chosen recording.
    '''
    data_file = None

    def translate_path(self, path):
        if path == "/data.json":
            return self.data_file
        return super().translate_path(path)

    def log_message(self, *args):
        # no log line for every request
        pass


def main():
    parser = argparse.ArgumentParser(description="3D playback in the browser")
    parser.add_argument("synced_dir", type=str, nargs="?", help="e.g. recordings/test/synced")
    args = parser.parse_args()

    # Interactive fallback
    if args.synced_dir is None:
        args.synced_dir = input("Synced folder (e.g. recordings/test/synced): ").strip()

    data_file = os.path.realpath(os.path.join(args.synced_dir, "poses3d.json"))
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run triangulate.py first.")
    Handler.data_file = data_file

    handler = functools.partial(Handler, directory=VIEWER_DIR)
    server = http.server.HTTPServer(("127.0.0.1", PORT), handler)

    print(f"viewer on http://localhost:{PORT} ")
    webbrowser.open(f"http://localhost:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# python3 view.py recordings/test/synced

if __name__ == "__main__":
    main()