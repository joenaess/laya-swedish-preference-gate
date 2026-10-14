"""Run the full pytest suite with coverage and record the results in `results/metrics.json`."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

from .utils import CACHE_DIR, ROOT, update_metrics


def main(argv=None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    junit, cov = CACHE_DIR / "junit.xml", CACHE_DIR / "coverage.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-p",
        "no:cacheprovider",
        "--cov=src/laya_gate",
        "--cov-report=term-missing",
        f"--cov-report=json:{cov}",
        f"--junitxml={junit}",
    ]
    code = subprocess.call(cmd, cwd=ROOT)

    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total = int(suite.get("tests"))
    failed, errors = int(suite.get("failures")), int(suite.get("errors"))
    skipped = int(suite.get("skipped"))
    skips = {}
    for case in suite.iter("testcase"):
        s = case.find("skipped")
        if s is not None:
            skips[f"{case.get('classname')}::{case.get('name')}"] = s.get("message", "")

    data = json.loads(cov.read_text())
    per_module = {
        path.split("/")[-1]: round(info["summary"]["percent_covered"], 1)
        for path, info in sorted(data["files"].items())
    }
    update_metrics(
        "tests",
        {
            "command": "pytest -v --cov=src/laya_gate (all markers; skips listed)",
            "total": total,
            "passed": total - failed - errors - skipped,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "skipped_tests": skips,
            "all_passed": failed == 0 and errors == 0,
            "coverage_percent": round(data["totals"]["percent_covered"], 1),
            "coverage_per_module": per_module,
        },
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
