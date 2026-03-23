import os
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    here = Path(__file__).resolve().parent
    load_env_file(here / ".env")

    token = os.getenv("HF_API_TOKEN")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    timeout = float(os.getenv("HF_TIMEOUT_SECONDS", "20"))

    if not token:
        print("Missing HF_API_TOKEN. Put it in Flask Deployed App/.env")
        return 2

    url = "https://router.huggingface.co/v1/chat/completions"
    prompt = "Give 3 quick steps to treat tomato leaf blight (generic guidance)."

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 160,
    }

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print("HTTP", resp.status)
            print(body)
            return 0
    except HTTPError as e:
        print("HTTPError", e.code)
        try:
            print(e.read().decode("utf-8", errors="replace"))
        except Exception:
            pass
        return 1
    except URLError as e:
        print("URLError", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
