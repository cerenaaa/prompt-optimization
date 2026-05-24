# Prompt Optimization Toolkit

[![CI](https://github.com/cerenaaa/prompt-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/prompt-optimization/actions)

Systematic prompt engineering framework: few-shot optimization, chain-of-thought injection, regression testing across prompt versions, and DSPy-style automatic prompt compilation.

## Problem

Prompts are fragile — small changes can significantly affect output quality. This toolkit treats prompts as code: version-controlled, tested, and optimized against a labeled evaluation set.

## Components

| Component | Purpose |
|---|---|
| `PromptVariantRunner` | A/B test multiple prompt versions against a dataset |
| `FewShotOptimizer` | Greedy selection of best few-shot examples per task |
| `ChainOfThoughtInjector` | Automatically adds CoT scaffolding to any prompt |
| `PromptRegressionSuite` | CI-style tests to catch prompt regressions on push |
| `PromptScorer` | LLM-judged quality scoring with rubrics |

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
python optimize.py --task classification --n_candidates 5
```

## Workflow

```
Dataset → Candidate Prompts → Parallel Eval → Score → Select Best → Regression Test
```
