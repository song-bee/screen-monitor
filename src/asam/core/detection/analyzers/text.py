"""
Text Content Analyzer

Uses local LLM (Ollama) to analyze text content for entertainment detection.
"""

import asyncio
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
            # Give time for connectors to close properly
            await asyncio.sleep(0.1)
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
            self.logger.debug("Skipping analysis - should_analyze returned False")
            return None

        if not self.session:
            self.logger.error("Text analyzer not initialized")
            return None

        analysis_start_time = datetime.now()

        try:
            # Step 1: Text preprocessing
            self.logger.debug("▶️  Step 1: ULTRA-VERBOSE TEXT PREPROCESSING")
            original_length = len(data.content)
            text_to_analyze = data.content[: self.max_text_length]
            truncated = original_length > self.max_text_length

            self.logger.info(
                f"✅ Text preprocessing COMPLETE - "
                f"Original: {original_length:,} chars, "
                f"Analyzing: {len(text_to_analyze):,} chars, "
                f"Truncated: {truncated}"
            )

            # Enhanced content analysis
            word_count = len(text_to_analyze.split())
            line_count = len(text_to_analyze.split("\n"))
            avg_word_length = len(text_to_analyze.replace(" ", "")) / max(word_count, 1)

            self.logger.debug(
                f"📊 Content statistics: {word_count} words, {line_count} lines, avg word length: {avg_word_length:.1f}"
            )

            # Show detailed content preview with line numbers
            preview_lines = text_to_analyze[:300].split("\n")[:5]
            self.logger.debug("📖 Content preview (first 5 lines):")
            for i, line in enumerate(preview_lines, 1):
                line_display = line[:80] + "..." if len(line) > 80 else line
                self.logger.debug(f"   {i:2d}: {repr(line_display)}")

            # Content type hints
            content_lower = text_to_analyze.lower()
            gaming_keywords = sum(
                1
                for word in ["game", "play", "level", "score", "win"]
                if word in content_lower
            )
            social_keywords = sum(
                1
                for word in ["like", "share", "follow", "post"]
                if word in content_lower
            )
            work_keywords = sum(
                1
                for word in ["code", "api", "tutorial", "documentation"]
                if word in content_lower
            )

            self.logger.debug(
                f"🎯 Content hints - Gaming: {gaming_keywords}, Social: {social_keywords}, Work: {work_keywords}"
            )

            # Step 2: Prompt creation
            self.logger.debug("▶️  Step 2: ULTRA-VERBOSE PROMPT CREATION")
            prompt_start_time = datetime.now()
            prompt = self._create_analysis_prompt(text_to_analyze)
            prompt_creation_time = (
                datetime.now() - prompt_start_time
            ).total_seconds() * 1000

            self.logger.info(
                f"✅ Prompt creation COMPLETE in {prompt_creation_time:.2f}ms"
            )
            self.logger.debug(
                f"📏 Prompt statistics: {len(prompt):,} chars, {len(prompt.split()):,} words"
            )

            # Show prompt structure analysis
            prompt_lines = prompt.split("\n")
            instruction_lines = len(
                [line for line in prompt_lines if line.strip().startswith("- ")]
            )
            example_lines = len(
                [line for line in prompt_lines if "example" in line.lower()]
            )

            self.logger.debug(
                f"📋 Prompt structure: {len(prompt_lines)} lines, {instruction_lines} instructions, {example_lines} examples"
            )

            # Show the actual prompt sections (more detailed)
            prompt_sections = prompt.split("\n\n")
            for i, section in enumerate(
                prompt_sections[:3], 1
            ):  # Show first 3 sections
                section_preview = (
                    section[:150] + "..." if len(section) > 150 else section
                )
                self.logger.debug(f"📄 Prompt section {i}: {repr(section_preview)}")

            # Step 3: LLM Query
            self.logger.debug("▶️  Step 3: ULTRA-VERBOSE LLM QUERY")
            llm_start_time = datetime.now()

            # Log query parameters
            self.logger.debug("🤖 LLM Configuration:")
            self.logger.debug(f"   Model: {self.model_name}")
            self.logger.debug(f"   Endpoint: {self.ollama_url}/api/generate")
            self.logger.debug("   Temperature: 0.1 (low for consistency)")
            self.logger.debug("   Format: JSON (structured output)")

            llm_response = await self._query_llm(prompt)
            llm_query_time = (datetime.now() - llm_start_time).total_seconds() * 1000

            self.logger.info(f"✅ LLM query COMPLETE in {llm_query_time:.1f}ms")

            # Ultra-detailed LLM response logging
            if llm_response:
                response_text = llm_response.get("response", "No response")
                self.logger.debug("📨 Raw LLM response analysis:")
                self.logger.debug(f"   Length: {len(response_text):,} characters")
                self.logger.debug(
                    f"   Lines: {len(response_text.split(chr(10)))} lines"
                )

                # Show full response in sections
                if len(response_text) > 500:
                    # Show first 250 chars and last 250 chars
                    self.logger.debug(
                        f"📄 LLM Response (first 250 chars): {repr(response_text[:250])}"
                    )
                    self.logger.debug(
                        f"📄 LLM Response (last 250 chars): {repr(response_text[-250:])}"
                    )
                else:
                    self.logger.debug(
                        f"📄 Complete LLM Response: {repr(response_text)}"
                    )

                # Log ultra-detailed LLM performance metadata
                if "eval_count" in llm_response:
                    self.logger.debug("🧠 LLM Performance:")
                    self.logger.debug(
                        f"   Tokens evaluated: {llm_response['eval_count']:,}"
                    )

                if "eval_duration" in llm_response:
                    eval_duration_ms = llm_response["eval_duration"] / 1_000_000
                    tokens_per_sec = (
                        llm_response.get("eval_count", 0) / (eval_duration_ms / 1000)
                        if eval_duration_ms > 0
                        else 0
                    )
                    self.logger.debug(
                        f"   Evaluation time: {eval_duration_ms:.1f}ms ({tokens_per_sec:.1f} tokens/sec)"
                    )

                if "total_duration" in llm_response:
                    total_duration_ms = llm_response["total_duration"] / 1_000_000
                    self.logger.debug(
                        f"   Total LLM duration: {total_duration_ms:.1f}ms"
                    )

                if "load_duration" in llm_response:
                    load_duration_ms = llm_response["load_duration"] / 1_000_000
                    self.logger.debug(f"   Model load time: {load_duration_ms:.1f}ms")

                if "prompt_eval_count" in llm_response:
                    self.logger.debug(
                        f"   Prompt tokens: {llm_response['prompt_eval_count']:,}"
                    )

                # Try to parse and preview JSON structure
                try:
                    parsed_response = json.loads(response_text)
                    self.logger.debug("🔍 Parsed JSON structure:")
                    for key, value in parsed_response.items():
                        value_preview = (
                            str(value)[:100] + "..."
                            if len(str(value)) > 100
                            else str(value)
                        )
                        self.logger.debug(f"   {key}: {repr(value_preview)}")
                except json.JSONDecodeError:
                    self.logger.debug(
                        "⚠️  Response is not valid JSON - will attempt parsing in next step"
                    )

            # Step 4: Response parsing
            self.logger.debug("▶️  Step 4: ULTRA-VERBOSE RESPONSE PARSING")
            parse_start_time = datetime.now()
            result = self._parse_llm_response(llm_response, data)
            parse_time = (datetime.now() - parse_start_time).total_seconds() * 1000

            self.logger.info(f"✅ Response parsing COMPLETE in {parse_time:.2f}ms")

            # Step 5: Final result logging with ultra-verbose details
            total_analysis_time = (
                datetime.now() - analysis_start_time
            ).total_seconds() * 1000

            if result:
                self.logger.info("🎉 TEXT ANALYSIS SUCCESSFUL!")
                self.logger.info("🎯 FINAL RESULT:")
                self.logger.info(f"   Category: {result.category.value.upper()}")
                self.logger.info(
                    f"   Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)"
                )
                self.logger.info(f"   Processing time: {total_analysis_time:.1f}ms")

                # Ultra-detailed evidence logging
                if result.evidence:
                    evidence = result.evidence
                    self.logger.info("🔍 ULTRA-VERBOSE EVIDENCE ANALYSIS:")

                    # LLM-specific evidence with detailed formatting
                    self.logger.info("   🤖 LLM Classification:")
                    self.logger.info(
                        f"      Raw Category: '{evidence.get('llm_category', 'unknown')}'"
                    )
                    self.logger.info(
                        f"      Subcategory: '{evidence.get('subcategory', 'unclear')}'"
                    )
                    self.logger.info(
                        f"      Model: {evidence.get('model_used', 'unknown')}"
                    )

                    # Keywords with detailed analysis
                    keywords = evidence.get("keywords", [])
                    self.logger.info(
                        f"   🔑 Keywords Analysis ({len(keywords)} total):"
                    )
                    if keywords:
                        # Group by keyword length and type
                        short_words = [k for k in keywords if len(k) <= 6]
                        medium_words = [k for k in keywords if 7 <= len(k) <= 12]
                        long_words = [k for k in keywords if len(k) > 12]

                        if short_words:
                            self.logger.info(f"      Short (≤6 chars): {short_words}")
                        if medium_words:
                            self.logger.info(f"      Medium (7-12): {medium_words}")
                        if long_words:
                            self.logger.info(f"      Long (>12): {long_words}")
                    else:
                        self.logger.info("      No keywords identified by LLM")

                    # Reasoning with sentence-by-sentence analysis
                    reasoning = evidence.get("reasoning", "No reasoning provided")
                    self.logger.info("   💭 LLM Reasoning Analysis:")
                    self.logger.info(f"      Length: {len(reasoning)} characters")

                    # Break down reasoning into sentences
                    sentences = [
                        s.strip() + "." for s in reasoning.split(".") if s.strip()
                    ]
                    self.logger.info(f"      Sentence count: {len(sentences)}")

                    for i, sentence in enumerate(
                        sentences[:6], 1
                    ):  # Show first 6 sentences
                        self.logger.info(f"      {i:2d}. {sentence}")

                    if len(sentences) > 6:
                        self.logger.info(
                            f"      ... plus {len(sentences) - 6} more sentences"
                        )

                    # Technical metadata with enhanced details
                    self.logger.debug("   🔧 Technical Metadata:")
                    self.logger.debug(
                        f"      Original text length: {evidence.get('text_length', 'unknown'):,}"
                    )
                    self.logger.debug(
                        f"      Text source: {evidence.get('text_source', 'unknown')}"
                    )
                    self.logger.debug(
                        f"      Processing timestamp: {datetime.now().isoformat()}"
                    )

                    # All evidence keys for debugging
                    all_keys = list(evidence.keys())
                    self.logger.debug(f"      All evidence fields: {all_keys}")

                # Ultra-detailed performance breakdown
                self.logger.info("📊 ULTRA-VERBOSE PERFORMANCE BREAKDOWN:")
                self.logger.info("   ⚡ Stage timings:")
                self.logger.info(f"      1. Preprocessing: {0:.1f}ms (instant)")
                self.logger.info(
                    f"      2. Prompt creation: {prompt_creation_time:.2f}ms"
                )
                self.logger.info(f"      3. LLM query: {llm_query_time:.1f}ms")
                self.logger.info(f"      4. Response parsing: {parse_time:.2f}ms")
                self.logger.info(f"      📈 TOTAL: {total_analysis_time:.1f}ms")

                # Efficiency metrics
                chars_per_ms = original_length / max(total_analysis_time, 0.1)
                self.logger.debug(f"   📈 Efficiency: {chars_per_ms:.1f} chars/ms")

            else:
                self.logger.error(
                    f"❌ TEXT ANALYSIS FAILED after {total_analysis_time:.1f}ms"
                )
                self.logger.error("   No detection result was generated")
                self.logger.error(
                    "   Check previous log entries for detailed error information"
                )

            return result

        except Exception as e:
            total_time = (datetime.now() - analysis_start_time).total_seconds() * 1000
            self.logger.error(
                f"💥 CRITICAL ERROR in text analysis after {total_time:.1f}ms"
            )
            self.logger.error(f"   Error message: {str(e)}")
            self.logger.error(f"   Error type: {type(e).__name__}")

            # Ultra-verbose error details
            import traceback

            tb_lines = traceback.format_exc().split("\n")
            self.logger.error("🐛 ULTRA-VERBOSE ERROR TRACEBACK:")
            for i, line in enumerate(tb_lines, 1):
                if line.strip():
                    self.logger.error(f"   {i:2d}: {line}")

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
