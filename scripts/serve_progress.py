"""Local research preview with HTTP byte ranges for synchronized video seeking."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()
        try:
            stream = path.open("rb")
        except OSError:
            return super().send_head()
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        partial = bool(match and (match[1] or match[2]))
        if partial:
            if match[1]:
                start = int(match[1]); end = min(int(match[2]), size-1) if match[2] else size-1
            else:
                start = max(0, size-int(match[2]))
            if start > end or start >= size:
                stream.close(); self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0"); self.end_headers(); return None
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(end-start+1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers(); stream.seek(start); self.remaining=end-start+1
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining:
                chunk=source.read(min(65536,self.remaining))
                if not chunk: break
                outputfile.write(chunk); self.remaining-=len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel an old range when the playhead moves.


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Progress site: http://127.0.0.1:{args.port}/progress/",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
