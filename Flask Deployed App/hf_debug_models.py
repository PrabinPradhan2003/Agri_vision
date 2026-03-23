import os
import json
import sys
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
    if not token:
        print("Missing HF_API_TOKEN")
        return 2

    req = Request(
        "https://router.huggingface.co/v1/models",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "Plant-Disease-Detection/1.0",
        },
        method="GET",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
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

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        print("No model data returned")
        return 1

    target_id = sys.argv[1].strip() if len(sys.argv) > 1 else None
    chosen = None
    if target_id:
        for it in items:
            if isinstance(it, dict) and it.get("id") == target_id:
                chosen = it
                break
        if not chosen:
            print("Model not found:", target_id)
            print("Tip: run hf_list_models.py to see available ids")
            return 1
    else:
        chosen = items[0]

    if isinstance(chosen, dict):
        print("keys:", sorted(chosen.keys()))
        print("id:", chosen.get("id"))
        providers = chosen.get("providers")
        if isinstance(providers, list):
            print("providers:")
            for p in providers:
                if isinstance(p, dict):
                    print(
                        "-",
                        p.get("provider"),
                        "status=",
                        p.get("status"),
                        "ctx=",
                        p.get("context_length"),
                    )
        else:
            print("providers: <missing>")
        print("sample:", json.dumps(chosen, ensure_ascii=False)[:1500])
    else:
        print("unexpected item type:", type(chosen))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
