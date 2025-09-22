"""
Text Content Analyzer

Uses local LLM (Ollama) to analyze text content for entertainment detection.
"""

import json
from datetime import datetime
from typing import Any, Optional

import aiohttp

from ..types import AnalysisType, ContentCategory, DetectionResult, TextContent
from .base import AnalyzerBase


class TextAnalyzer(AnalyzerBase):
    """Analyzes text content using local LLM for entertainment detection"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        self.model_name = self.config.get("model_name", "llama3.2:3b")
        self.max_text_length = self.config.get("max_text_length", 4000)
        self.session: Optional[aiohttp.ClientSession] = None

    @property
    def analyzer_type(self) -> AnalysisType:
        return AnalysisType.TEXT

    async def initialize(self) -> bool:
        """Initialize the text analyzer and check LLM availability"""
        if not await super().initialize():
            return False

        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

        # Check if Ollama is running and model is available
        try:
            await self._check_ollama_health()
            await self._ensure_model_available()
            self.logger.info("Text analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize text analyzer: {e}")
            await self.cleanup()
            return False

    async def cleanup(self) -> None:
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        await super().cleanup()

    async def analyze(self, data: TextContent) -> Optional[DetectionResult]:
        """
        Analyze text content using LLM

        Args:
            data: TextContent object with text to analyze

        Returns:
            DetectionResult with confidence and category
        """
        if not self.should_analyze(data):
            return None

        if not self.session:
            self.logger.error("Text analyzer not initialized")
            return None

        try:
            # Truncate text if too long
            text_to_analyze = data.content[: self.max_text_length]

            # Create analysis prompt
            prompt = self._create_analysis_prompt(text_to_analyze)

            # Query LLM
            llm_response = await self._query_llm(prompt)

            # Parse response and create result
            return self._parse_llm_response(llm_response, data)

        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return None

    def should_analyze(self, data: TextContent) -> bool:
        """Check if text content should be analyzed"""
        if not super().should_analyze(data):
            return False

        # Skip if text is too short or empty
        if not data.content or len(data.content.strip()) < 10:
            return False

        return True

    async def _check_ollama_health(self) -> None:
        """Check if Ollama service is running"""
        try:
            async with self.session.get(f"{self.ollama_url}/api/version") as response:
                if response.status != 200:
                    raise Exception(f"Ollama health check failed: {response.status}")
        except Exception as e:
            raise Exception(f"Cannot connect to Ollama at {self.ollama_url}: {e}")

    async def _ensure_model_available(self) -> None:
        """Ensure the required model is available"""
        try:
            async with self.session.get(f"{self.ollama_url}/api/tags") as response:
                if response.status != 200:
                    raise Exception(f"Cannot list models: {response.status}")

                data = await response.json()
                available_models = [model["name"] for model in data.get("models", [])]

                if self.model_name not in available_models:
                    self.logger.warning(
                        f"Model {self.model_name} not found. Attempting to pull..."
                    )
                    await self._pull_model()

        except Exception as e:
            raise Exception(f"Error checking model availability: {e}")

    async def _pull_model(self) -> None:
        """Pull the required model from Ollama"""
        try:
            payload = {"name": self.model_name}
            async with self.session.post(
                f"{self.ollama_url}/api/pull", json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Model pull failed: {response.status}")

                # Wait for model to be pulled (this is a streaming response)
                async for line in response.content:
                    if line:
                        try:
                            status = json.loads(line.decode())
                            if status.get("status") == "success":
                                self.logger.info(
                                    f"Model {self.model_name} pulled successfully"
                                )
                                return
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            raise Exception(f"Error pulling model: {e}")

    def _create_analysis_prompt(self, text: str) -> str:
        """Create analysis prompt for the LLM"""
        return f"""You are an AI assistant that analyzes text content to determine if it represents entertainment/recreational activities or productive work.

ENTERTAINMENT INDICATORS (classify as "entertainment" with HIGH confidence 0.8+):
- Gaming: game titles, scores, achievements, gaming forums, Twitch, Steam
- Video streaming: YouTube entertainment, Netflix, movies, TV shows, TikTok, short videos
- Social media leisure: Instagram posts, Facebook timeline, Twitter social content, Reddit memes
- Fiction/novels: story content, character names, fictional narratives
- Celebrity/gossip: entertainment news, celebrity social media, tabloid content
- Leisure shopping: non-essential items, entertainment products, hobby materials

PRODUCTIVE INDICATORS (classify as "productive" with HIGH confidence 0.8+):
- Work documents: emails, reports, spreadsheets, project management, presentations
- Technical content: code, documentation, API references, technical tutorials
- Educational: academic papers, learning materials, certification content, courses
- Professional development: skill building, career-related content, training
- News/research: factual news articles, data analysis, research papers
- Work tools: professional software interfaces, productivity apps

MIXED/AMBIGUOUS (classify as "unknown" with LOWER confidence 0.3-0.6):
- Context unclear from text alone
- Could be either entertainment or work depending on context
- Window titles without clear content
- Generic interface elements

TEXT TO ANALYZE:
{text}

INSTRUCTIONS:
1. Focus on CONTENT TYPE rather than platform (YouTube can be educational or entertainment)
2. Consider INTENT (learning vs leisure consumption)
3. Look for specific keywords and context clues
4. Be confident in clear cases, conservative in ambiguous ones

Respond ONLY in valid JSON:
{{
    "category": "entertainment|productive|unknown",
    "confidence": 0.0-1.0,
    "reasoning": "specific evidence from text",
    "keywords": ["specific", "relevant", "terms"],
    "subcategory": "gaming|video|social|work|education|news|unclear"
}}"""

    async def _query_llm(self, prompt: str) -> dict[str, Any]:
        """Query the LLM with the analysis prompt"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,  # Low temperature for consistent results
                "top_p": 0.9,
                "num_ctx": 4096,
            },
        }

        async with self.session.post(
            f"{self.ollama_url}/api/generate", json=payload
        ) as response:
            if response.status != 200:
                raise Exception(f"LLM query failed: {response.status}")

            result = await response.json()
            return result

    def _parse_llm_response(
        self, llm_response: dict[str, Any], original_data: TextContent
    ) -> DetectionResult:
        """Parse LLM response and create DetectionResult"""
        try:
            # Parse the JSON response from LLM
            response_text = llm_response.get("response", "{}")
            analysis = json.loads(response_text)

            # Map category
            category_mapping = {
                "entertainment": ContentCategory.ENTERTAINMENT,
                "productive": ContentCategory.PRODUCTIVE,
                "unknown": ContentCategory.UNKNOWN,
            }

            llm_category = analysis.get("category", "unknown").lower()
            category = category_mapping.get(llm_category, ContentCategory.UNKNOWN)

            # Get confidence
            confidence = float(analysis.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range

            # Build evidence
            evidence = {
                "llm_category": llm_category,
                "subcategory": analysis.get("subcategory", "unclear"),
                "reasoning": analysis.get("reasoning", ""),
                "keywords": analysis.get("keywords", []),
                "text_length": len(original_data.content),
                "text_source": original_data.source,
                "model_used": self.model_name,
            }

            return DetectionResult(
                analyzer_type=self.analyzer_type,
                confidence=confidence,
                category=category,
                evidence=evidence,
                timestamp=datetime.now(),
                metadata={"original_text_length": len(original_data.content)},
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            # Return low-confidence unknown result as fallback
            return DetectionResult(
                analyzer_type=self.analyzer_type,
                confidence=0.1,
                category=ContentCategory.UNKNOWN,
                evidence={
                    "error": str(e),
                    "raw_response": llm_response.get("response", ""),
                    "text_source": original_data.source,
                },
                timestamp=datetime.now(),
                metadata={"parse_error": True},
            )
