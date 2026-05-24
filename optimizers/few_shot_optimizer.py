"""
Greedy few-shot example selector.
Finds the subset of examples that maximizes eval score.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
import anthropic


@dataclass
class FewShotExample:
    input_text: str
    output: str

    def format(self) -> str:
        return f"Input: {self.input_text}\nOutput: {self.output}"


class FewShotOptimizer:
    """
    Greedy forward selection: iteratively adds the example that most
    improves performance on the eval set.
    Inspired by DSPy's BootstrapFewShot optimizer.
    """

    def __init__(self, max_examples: int = 5, model: str = "claude-sonnet-4-20250514"):
        self.max_examples = max_examples
        self.model = model
        self.client = anthropic.Anthropic()

    def _evaluate(
        self,
        examples: list[FewShotExample],
        eval_set: list[dict],
        task_instruction: str,
    ) -> float:
        few_shot_block = "\n\n".join(e.format() for e in examples)
        system = f"{task_instruction}\n\nExamples:\n{few_shot_block}" if examples else task_instruction
        scores = []
        for sample in eval_set[:20]:  # cap at 20 for speed
            resp = self.client.messages.create(
                model=self.model, max_tokens=256, system=system,
                messages=[{"role": "user", "content": f"Input: {sample['input']}"}],
            )
            output = resp.content[0].text.strip()
            expected_tokens = set(sample["expected"].lower().split())
            out_tokens = set(output.lower().split())
            scores.append(len(out_tokens & expected_tokens) / (len(expected_tokens) + 1e-9))
        return sum(scores) / len(scores) if scores else 0

    def optimize(
        self,
        candidate_examples: list[FewShotExample],
        eval_set: list[dict],
        task_instruction: str,
    ) -> list[FewShotExample]:
        selected: list[FewShotExample] = []
        remaining = list(candidate_examples)
        baseline = self._evaluate([], eval_set, task_instruction)
        print(f"Baseline score (no examples): {baseline:.3f}")

        for _ in range(self.max_examples):
            best_score, best_example = baseline, None
            random.shuffle(remaining)
            for candidate in remaining[:15]:  # check up to 15 per round
                trial = selected + [candidate]
                score = self._evaluate(trial, eval_set, task_instruction)
                if score > best_score:
                    best_score, best_example = score, candidate
            if best_example is None:
                print("No further improvement — stopping early")
                break
            selected.append(best_example)
            remaining.remove(best_example)
            baseline = best_score
            print(f"  Added example #{len(selected)}: score={best_score:.3f}")

        return selected