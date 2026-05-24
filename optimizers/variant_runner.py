"""
Prompt variant A/B runner.
Evaluates multiple prompt templates against a labeled dataset in parallel.
"""
from __future__ import annotations
import json
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable
import anthropic


@dataclass
class PromptVariant:
    name: str
    system: str
    template: str       # use {input} as placeholder for the user input
    metadata: dict = field(default_factory=dict)

    def render(self, input_text: str) -> str:
        return self.template.format(input=input_text)


@dataclass
class EvalSample:
    input_text: str
    expected_output: str
    metadata: dict = field(default_factory=dict)


@dataclass
class VariantResult:
    variant_name: str
    score: float
    latency_ms: float
    outputs: list[str]
    errors: int


class PromptVariantRunner:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_workers: int = 4):
        self.model = model
        self.client = anthropic.Anthropic()
        self.max_workers = max_workers

    def _call(self, variant: PromptVariant, sample: EvalSample) -> tuple[str, float]:
        import time
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=variant.system,
            messages=[{"role": "user", "content": variant.render(sample.input_text)}],
        )
        ms = (time.perf_counter() - t0) * 1000
        return resp.content[0].text.strip(), ms

    def _score_output(self, output: str, expected: str) -> float:
        """Simple token-overlap score. Override with LLM judge for production."""
        out_tokens = set(output.lower().split())
        exp_tokens = set(expected.lower().split())
        if not exp_tokens:
            return 0.0
        return len(out_tokens & exp_tokens) / len(exp_tokens)

    def evaluate_variant(self, variant: PromptVariant, samples: list[EvalSample]) -> VariantResult:
        scores, latencies, outputs, errors = [], [], [], 0
        for sample in samples:
            try:
                output, ms = self._call(variant, sample)
                score = self._score_output(output, sample.expected_output)
                scores.append(score)
                latencies.append(ms)
                outputs.append(output)
            except Exception as e:
                errors += 1
                outputs.append(f"ERROR: {e}")

        return VariantResult(
            variant_name=variant.name,
            score=sum(scores) / len(scores) if scores else 0,
            latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            outputs=outputs,
            errors=errors,
        )

    def run(self, variants: list[PromptVariant], samples: list[EvalSample]) -> list[VariantResult]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.evaluate_variant, v, samples): v for v in variants}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"  {result.variant_name}: score={result.score:.3f} | "
                      f"latency={result.latency_ms:.0f}ms | errors={result.errors}")

        return sorted(results, key=lambda r: r.score, reverse=True)