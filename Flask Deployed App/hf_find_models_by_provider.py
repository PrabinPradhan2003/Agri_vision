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

    wanted = [p.strip() for p in (sys.argv[1:] or ["hf-inference", "groq", "cerebras", "fireworks", "hyperbolic"]) if p.strip()]

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
    if not isinstance(items, list):
        print("Unexpected response")
        return 1

    matches = []
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = it.get("id")
        providers = it.get("providers")
        if not isinstance(mid, str) or not isinstance(providers, list):
            continue
        prov_names = [p.get("provider") for p in providers if isinstance(p, dict) and isinstance(p.get("provider"), str) and p.get("status") == "live"]
        for w in wanted:
            if w in prov_names:
                matches.append((mid, w, prov_names))
                break

    print("Wanted providers:", ", ".join(wanted))
    print("Matches:", len(matches))
    for mid, w, provs in matches[:80]:
        print(f"{mid}  -> has {w}  (all live providers: {', '.join(provs)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
