"""
Mock models for testing and demonstration.
"""

import random
import hashlib
from typing import List

from .base import BaseModel, ModelConfig, ModelRegistry


@ModelRegistry.register("mock")
class MockModel(BaseModel):
    """Mock модель для демонстрации без реальных моделей."""
    
    TEMPLATES = {
        "verbose": [
            "This post discusses {topic}. The main point is that the author wants to share their experience about {detail}. In conclusion, {conclusion}.",
            "The author is talking about {topic} and explains that {detail}. They conclude that {conclusion}.",
            "Here we have a post about {topic}. Key insight: {detail}. Final thought: {conclusion}.",
        ],
        "concise": [
            "{topic}: {detail}. {conclusion}.",
            "Author discusses {topic}, noting {detail}.",
            "Post about {topic} - {conclusion}.",
        ],
        "balanced": [
            "Post about {topic}. Main point: {detail}. Takeaway: {conclusion}.",
            "The author shares thoughts on {topic}, specifically {detail}.",
            "{topic} discussion: {detail}. Worth noting: {conclusion}.",
        ]
    }
    
    TOPICS = [
        "relationships", "work", "family", "technology", "gaming",
        "cooking", "travel", "health", "finance", "education"
    ]
    
    DETAILS = [
        "things aren't always what they seem",
        "communication is key",
        "patience pays off",
        "first impressions matter",
        "context is important"
    ]
    
    CONCLUSIONS = [
        "situation improved", "lesson learned", "still figuring it out",
        "happy ending", "work in progress", "unexpected outcome"
    ]
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.strength = config.strength
        self.style = config.style
        self._rng = random.Random(hash(config.name))
    
    def generate(self, prompt: str) -> str:
        """Generate deterministic mock response based on prompt."""
        # Use prompt hash for deterministic but varied responses
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        self._rng.seed(prompt_hash + hash(self.name))
        
        template = self._rng.choice(self.TEMPLATES[self.style])
        
        response = template.format(
            topic=self._rng.choice(self.TOPICS),
            detail=self._rng.choice(self.DETAILS),
            conclusion=self._rng.choice(self.CONCLUSIONS)
        )
        
        return response
    
    def is_available(self) -> bool:
        return True
    
    def get_expected_score(self) -> float:
        """Ожидаемый скор (для тестирования)."""
        return self.strength


def create_mock_models(
    names: List[str] = None,
    styles: List[str] = None,
    strengths: List[float] = None
) -> dict[str, MockModel]:
    """Создать набор mock моделей."""
    if names is None:
        names = ["model_A", "model_B", "model_C", "model_D"]
    
    if styles is None:
        styles = ["verbose", "concise", "balanced", "balanced"]
    
    if strengths is None:
        strengths = [0.6, 0.75, 0.85, 0.7]
    
    return {
        name: MockModel(ModelConfig(
            name=name,
            model_type="mock",
            strength=strength,
            style=style
        ))
        for name, style, strength in zip(names, styles, strengths)
    }
