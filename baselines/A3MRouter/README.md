# A3M Router

Adaptive Memory Multi-Model Router for LLMRouterBench.

**Reference:** https://github.com/Das-rebel/a3m-router
**NPM:** `npm install adaptive-memory-multi-model-router@2.14.58`

## Key Features

1. **Parallel Multi-LLM Execution** - Execute multiple providers simultaneously
2. **Confidence-Weighted Voting** - Shapley value-based credit assignment
3. **Complexity-Aware Routing** - Estimates prompt complexity and routes accordingly
4. **Thompson Sampling** - Bayesian exploration/exploitation balance

## Performance

| Metric | Value |
|--------|-------|
| Exact Tier Match | 67% |
| ±1 Tier Accuracy | 96% |
| Cost Savings | 62.9% |
| Robustness | 0.8524 |

## Usage

```python
from baselines.A3MRouter import A3MRouter

router = A3MRouter(model_pool=["gpt-4", "qwen-2.5-72b"])
selected_model = router.route("Your prompt here")
```
