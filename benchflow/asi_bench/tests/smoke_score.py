"""Score an ASI result using the adapter's bundled evaluator and rebuilt image."""
import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.asi_bench.scorer import score_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', action='store_true', required=True)
    for name in ('prepared', 'result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    report = asyncio.run(score_result(args.prepared, args.result, args.output,
                                     timeout=args.timeout))
    print(json.dumps({'status': report['evaluation_status'], 'score': report['score']}))


if __name__ == '__main__':
    main()
