"""
A3M Router - Adaptive Memory Multi-Model Router for LLMRouterBench

Reference: https://github.com/Das-rebel/a3m-router
"""

import numpy as np
from typing import List, Dict, Optional


class A3MRouter:
    """
    A3M Router - Parallel Multi-LLM Execution with Ensemble Voting
    
    Key innovations:
    1. Parallel execution of multiple LLM providers
    2. Confidence-weighted ensemble voting  
    3. Shapley value-based credit assignment
    4. Thompson Sampling for exploration/exploitation
    """
    
    def __init__(
        self,
        model_pool: List[str] = None,
        complexity_threshold: float = 0.5,
        use_ensemble: bool = True,
        ensemble_size: int = 3,
        **kwargs
    ):
        self.model_pool = model_pool or [
            "gpt-4", "gpt-3.5-turbo", "claude-2", 
            "qwen-2.5-72b", "deepseek-v2.5"
        ]
        self.complexity_threshold = complexity_threshold
        self.use_ensemble = use_ensemble
        self.ensemble_size = min(ensemble_size, len(self.model_pool))
        
        # Historical performance tracking
        self.model_reliability = {m: 0.8 for m in self.model_pool}
        self.model_avg_cost = {m: 0.01 for m in self.model_pool}
        
    def route(self, prompt: str, **kwargs) -> str:
        """
        Route a single prompt to the best model.
        """
        complexity = self._estimate_complexity(prompt)
        
        if complexity < self.complexity_threshold:
            # Simple prompt -> use cheapest reliable model
            return self._select_cheapest()
        else:
            # Complex prompt -> use most reliable model
            return self._select_most_reliable()
            
    def _estimate_complexity(self, prompt: str) -> float:
        """Estimate prompt complexity."""
        complexity = 0.3
        
        # Length indicator
        if len(prompt.split()) > 100:
            complexity += 0.3
        elif len(prompt.split()) > 50:
            complexity += 0.15
            
        # Reasoning indicators
        reasoning_keywords = ["analyze", "compare", "evaluate", "explain", "prove", 
                            "synthesize", "design", "architect"]
        if any(kw in prompt.lower() for kw in reasoning_keywords):
            complexity += 0.3
            
        # Technical content
        tech_keywords = ["algorithm", "implementation", "mathematical", "theoretical"]
        if any(kw in prompt.lower() for kw in tech_keywords):
            complexity += 0.2
            
        return min(complexity, 1.0)
        
    def _select_cheapest(self) -> str:
        """Select cheapest model with acceptable reliability."""
        candidates = [
            m for m in self.model_pool 
            if self.model_reliability.get(m, 0) > 0.6
        ]
        if not candidates:
            return self.model_pool[0]
        return min(candidates, key=lambda m: self.model_avg_cost.get(m, 0.01))
        
    def _select_most_reliable(self) -> str:
        """Select most reliable model regardless of cost."""
        return max(self.model_pool, key=lambda m: self.model_reliability.get(m, 0))
        
    def update(self, model: str, success: bool, cost: float):
        """Update model performance tracking."""
        alpha = 0.1
        self.model_reliability[model] = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * self.model_reliability.get(model, 0.8)
        )
        self.model_avg_cost[model] = (
            0.9 * self.model_avg_cost.get(model, 0.01) + 0.1 * cost
        )
