from __future__ import annotations

import argparse
import json

from backend.climate.service import ClimateService


def main() -> None:
    parser = argparse.ArgumentParser(description="Train climate ML models")
    parser.add_argument("--refresh", action="store_true", help="Retrain even if saved artifacts already exist")
    args = parser.parse_args()

    service = ClimateService()
    result = service.train(force=args.refresh)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
