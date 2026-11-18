import json
import logging
import re
from typing import Any, Dict, List, Optional
from groq import AsyncGroq

logger = logging.getLogger(__name__)


class GroqLLMClient:
    """
    Groq API Client targeting `openai/gpt-oss-120b` for fast, structured agentic reasoning:
    1. Goal Decomposition & Planning Trace
    2. Self-Correction & Root-Cause Diagnosis when a tool fails
    3. Final Structured Report & Self-Evaluation Synthesis
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b"):
        self.api_key = (
            api_key
            if api_key and api_key != "your_groq_api_key_here"
            else None
        )
        self.model = model
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    @property
    def is_live_groq(self) -> bool:
        return self.client is not None

    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return self._extract_json(content)
        except Exception as exc:
            logger.warning(f"Groq JSON completion warning ({self.model}): {exc}")
            # Try without response_format in case reasoning model emits thinking tokens first
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt + "\nRespond ONLY with a valid JSON object.",
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=3000,
                )
                content = response.choices[0].message.content or "{}"
                return self._extract_json(content)
            except Exception as inner_exc:
                logger.error(f"Groq fallback completion failed: {inner_exc}")
                return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.strip()
        # Strip <think>...</think> if present
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        # Strip markdown fences
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
            if match:
                cleaned = match.group(1)
        try:
            return json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except Exception:
                    return None
        return None
