"""
Prompt regression test suite.
Runs on CI to catch regressions when prompts or model versions change.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable
import anthropic


@dataclass
class RegressionTest:
    name: str
    input_text: str
    expected_contains: list[str]     # output must contain these strings
    expected_not_contains: list[str] # output must NOT contain these
    min_length: int = 10


class PromptRegressionSuite:
    def __init__(self, system_prompt: str, model: str = "claude-sonnet-4-20250514"):
        self.system = system_prompt
        self.model = model
        self.client = anthropic.Anthropic()
        self.tests: list[RegressionTest] = []

    def add_test(self, test: RegressionTest):
        self.tests.append(test)
        return self

    def _run_test(self, test: RegressionTest) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=512, system=self.system,
            messages=[{"role": "user", "content": test.input_text}],
        )
        output = resp.content[0].text.strip()
        failures = []
        for phrase in test.expected_contains:
            if phrase.lower() not in output.lower():
                failures.append(f"Missing expected phrase: '{phrase}'")
        for phrase in test.expected_not_contains:
            if phrase.lower() in output.lower():
                failures.append(f"Unexpected phrase found: '{phrase}'")
        if len(output) < test.min_length:
            failures.append(f"Output too short: {len(output)} < {test.min_length}")
        return {"test": test.name, "passed": not failures, "failures": failures, "output": output[:200]}

    def run(self) -> dict:
        results = [self._run_test(t) for t in self.tests]
        passed = sum(1 for r in results if r["passed"])
        print(f"\nPrompt Regression Suite: {passed}/{len(results)} passed")
        for r in results:
            status = "✓" if r["passed"] else "✗"
            print(f"  {status} {r['test']}")
            for f in r["failures"]:
                print(f"      → {f}")
        return {"passed": passed, "total": len(results), "results": results}