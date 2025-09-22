"""
Confidence Aggregation Engine

Aggregates results from multiple analyzers and makes final decisions.
"""

import logging
from datetime import datetime
from statistics import mean, median
from typing import Any, Optional

from .types import (
    ActionType,
    AggregatedResult,
    AnalysisType,
    ContentCategory,
    DetectionResult,
)


class ConfidenceAggregator:
    """Aggregates confidence scores from multiple analyzers"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Confidence thresholds for actions
        self.block_threshold = self.config.get("block_threshold", 0.85)
        self.warn_threshold = self.config.get("warn_threshold", 0.65)
        self.log_threshold = self.config.get("log_threshold", 0.45)

        # Analyzer weights (how much each analyzer contributes to final decision)
        self.analyzer_weights = self.config.get(
            "analyzer_weights",
            {
                AnalysisType.TEXT.value: 0.4,  # High weight - LLM is primary
                AnalysisType.VISION.value: 0.25,  # Medium weight - visual analysis
                AnalysisType.PROCESS.value: 0.25,  # Medium weight - process monitoring
                AnalysisType.NETWORK.value: 0.1,  # Lower weight - network analysis
            },
        )

        # Category confidence requirements (minimum confidence to classify as category)
        self.category_thresholds = self.config.get(
            "category_thresholds",
            {
                ContentCategory.GAMING.value: 0.6,
                ContentCategory.VIDEO_STREAMING.value: 0.5,
                ContentCategory.SOCIAL_MEDIA.value: 0.4,
                ContentCategory.ENTERTAINMENT.value: 0.5,
            },
        )

        # Consensus requirements (how many analyzers must agree)
        self.require_consensus = self.config.get("require_consensus", False)
        self.min_consensus_analyzers = self.config.get("min_consensus_analyzers", 2)

        # Historical context
        self.enable_temporal_analysis = self.config.get(
            "enable_temporal_analysis", True
        )
        self.recent_results: list[AggregatedResult] = []
        self.max_history_size = self.config.get("max_history_size", 10)

    def aggregate(
        self, detection_results: list[DetectionResult], analysis_start_time: datetime
    ) -> AggregatedResult:
        """Legacy aggregation method - delegates to aggregate_with_rules with empty rules"""
        empty_rules_decision = {
            "action": ActionType.LOG_ONLY,
            "confidence": 0.0,
            "category": ContentCategory.UNKNOWN,
            "reasoning": "Legacy aggregation",
            "matched_rules": [],
            "analyzer_breakdown": {},
            "consecutive_count": 0,
        }
        return self.aggregate_with_rules(
            detection_results, analysis_start_time, empty_rules_decision
        )

    def aggregate_with_rules(
        self,
        detection_results: list[DetectionResult],
        analysis_start_time: datetime,
        rules_decision: dict[str, Any],
    ) -> AggregatedResult:
        """
        Aggregate detection results from multiple analyzers with advanced rules

        Args:
            detection_results: List of detection results from analyzers
            analysis_start_time: When the analysis started
            rules_decision: Decision from the advanced rules engine

        Returns:
            AggregatedResult with final decision
        """
        if not detection_results:
            return self._create_no_detection_result(analysis_start_time)

        # Filter out invalid results
        valid_results = [r for r in detection_results if self._is_valid_result(r)]
        if not valid_results:
            return self._create_no_detection_result(analysis_start_time)

        # Use rules engine decision if available, otherwise fall back to legacy logic
        if rules_decision and rules_decision.get("confidence", 0) > 0:
            # Use rules engine decision
            weighted_confidence = rules_decision["confidence"]
            primary_category = rules_decision["category"]
            recommended_action = rules_decision["action"]
            self.logger.debug(
                f"Using rules engine decision: {rules_decision['reasoning']}"
            )
        else:
            # Fall back to legacy aggregation logic
            # Calculate weighted confidence
            weighted_confidence = self._calculate_weighted_confidence(valid_results)

            # Determine primary category
            primary_category = self._determine_primary_category(
                valid_results, weighted_confidence
            )

            # Apply consensus requirements if enabled
            if self.require_consensus:
                consensus_result = self._apply_consensus_logic(
                    valid_results, weighted_confidence, primary_category
                )
                if consensus_result:
                    weighted_confidence, primary_category = consensus_result

            # Apply temporal analysis if enabled
            if self.enable_temporal_analysis:
                weighted_confidence = self._apply_temporal_analysis(
                    weighted_confidence, primary_category
                )

            # Determine recommended action
            recommended_action = self._determine_action(
                weighted_confidence, primary_category
            )

        # Calculate analysis duration
        analysis_duration = int(
            (datetime.now() - analysis_start_time).total_seconds() * 1000
        )

        # Create final result
        aggregated_result = AggregatedResult(
            overall_confidence=weighted_confidence,
            primary_category=primary_category,
            recommended_action=recommended_action,
            individual_results=valid_results,
            timestamp=datetime.now(),
            analysis_duration_ms=analysis_duration,
        )

        # Update history
        self._update_history(aggregated_result)

        self.logger.debug(
            f"Aggregated result: {primary_category.value} "
            f"(confidence: {weighted_confidence:.3f}, action: {recommended_action.value})"
        )

        return aggregated_result

    def _is_valid_result(self, result: DetectionResult) -> bool:
        """Check if a detection result is valid for aggregation"""
        if not result:
            return False

        # Check confidence bounds
        if not 0.0 <= result.confidence <= 1.0:
            self.logger.warning(f"Invalid confidence: {result.confidence}")
            return False

        # Check required fields
        if not result.analyzer_type or not result.category:
            self.logger.warning("Missing required fields in detection result")
            return False

        return True

    def _calculate_weighted_confidence(self, results: list[DetectionResult]) -> float:
        """Calculate weighted confidence score from all analyzers"""
        if not results:
            return 0.0

        # Group results by analyzer type
        analyzer_results = {}
        for result in results:
            analyzer_type = result.analyzer_type.value
            if analyzer_type not in analyzer_results:
                analyzer_results[analyzer_type] = []
            analyzer_results[analyzer_type].append(result)

        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0

        for analyzer_type, analyzer_results_list in analyzer_results.items():
            # Get weight for this analyzer
            weight = self.analyzer_weights.get(analyzer_type, 0.1)

            # Use maximum confidence if multiple results from same analyzer
            max_confidence = max(r.confidence for r in analyzer_results_list)

            # Weight entertainment categories higher than productive/unknown
            category_boost = self._get_category_confidence_boost(analyzer_results_list)
            adjusted_confidence = min(max_confidence + category_boost, 1.0)

            weighted_sum += adjusted_confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        base_confidence = weighted_sum / total_weight

        # Apply additional confidence adjustments
        final_confidence = self._apply_confidence_adjustments(base_confidence, results)

        return max(0.0, min(final_confidence, 1.0))

    def _get_category_confidence_boost(self, results: list[DetectionResult]) -> float:
        """Get confidence boost based on detected categories"""
        entertainment_categories = {
            ContentCategory.GAMING,
            ContentCategory.VIDEO_STREAMING,
            ContentCategory.ENTERTAINMENT,
            ContentCategory.SOCIAL_MEDIA,
        }

        # Boost confidence if entertainment categories are detected
        has_entertainment = any(r.category in entertainment_categories for r in results)
        return 0.1 if has_entertainment else 0.0

    def _apply_confidence_adjustments(
        self, base_confidence: float, results: list[DetectionResult]
    ) -> float:
        """Apply additional confidence adjustments"""
        adjusted_confidence = base_confidence

        # Multi-analyzer agreement boost
        unique_analyzers = len({r.analyzer_type for r in results})
        if unique_analyzers >= 3:
            adjusted_confidence += 0.1
        elif unique_analyzers >= 2:
            adjusted_confidence += 0.05

        # High individual confidence boost
        max_individual_confidence = max(r.confidence for r in results)
        if max_individual_confidence > 0.9:
            adjusted_confidence += 0.05

        # Evidence quality assessment
        evidence_quality = self._assess_evidence_quality(results)
        adjusted_confidence += evidence_quality * 0.1

        return adjusted_confidence

    def _assess_evidence_quality(self, results: list[DetectionResult]) -> float:
        """Assess the quality of evidence across all results"""
        quality_score = 0.0

        for result in results:
            evidence = result.evidence or {}

            # Score based on evidence richness
            evidence_richness = len(evidence) / 10.0  # Normalize by expected keys
            quality_score += min(evidence_richness, 1.0)

            # Bonus for specific high-quality evidence types
            if "llm_category" in evidence:  # LLM analysis
                quality_score += 0.2
            if "motion_ratio" in evidence:  # Motion detection
                quality_score += 0.1
            if "gaming_processes" in evidence:  # Process detection
                quality_score += 0.1

        # Average quality across all results
        return quality_score / len(results) if results else 0.0

    def _determine_primary_category(
        self, results: list[DetectionResult], overall_confidence: float
    ) -> ContentCategory:
        """Determine the primary content category"""
        if not results:
            return ContentCategory.UNKNOWN

        # Count category votes, weighted by confidence
        category_scores = {}

        for result in results:
            category = result.category
            # Weight by both result confidence and analyzer weight
            analyzer_weight = self.analyzer_weights.get(result.analyzer_type.value, 0.1)
            score = result.confidence * analyzer_weight

            if category in category_scores:
                category_scores[category] = max(category_scores[category], score)
            else:
                category_scores[category] = score

        # Get category with highest weighted score
        if not category_scores:
            return ContentCategory.UNKNOWN

        best_category = max(category_scores.keys(), key=lambda x: category_scores[x])
        best_score = category_scores[best_category]

        # Check if category meets minimum threshold
        category_threshold = self.category_thresholds.get(best_category.value, 0.5)
        if best_score < category_threshold:
            return ContentCategory.UNKNOWN

        return best_category

    def _apply_consensus_logic(
        self,
        results: list[DetectionResult],
        confidence: float,
        category: ContentCategory,
    ) -> Optional[tuple[float, ContentCategory]]:
        """Apply consensus requirements to decision"""
        if len(results) < self.min_consensus_analyzers:
            # Not enough analyzers for consensus
            return max(confidence * 0.5, 0.1), ContentCategory.UNKNOWN

        # Count how many analyzers agree on entertainment vs productive
        entertainment_categories = {
            ContentCategory.GAMING,
            ContentCategory.VIDEO_STREAMING,
            ContentCategory.ENTERTAINMENT,
            ContentCategory.SOCIAL_MEDIA,
        }

        entertainment_votes = sum(
            1 for r in results if r.category in entertainment_categories
        )

        total_analyzers = len(results)
        consensus_ratio = entertainment_votes / total_analyzers

        # Require majority for high-confidence entertainment classification
        if category in entertainment_categories:
            if consensus_ratio < 0.5:  # Less than 50% agreement
                return confidence * 0.6, ContentCategory.UNKNOWN
            elif consensus_ratio < 0.7:  # 50-70% agreement
                return confidence * 0.8, category

        return None  # Keep original decision

    def _apply_temporal_analysis(
        self, confidence: float, category: ContentCategory
    ) -> float:
        """Apply temporal analysis based on recent detection history"""
        if len(self.recent_results) < 2:
            return confidence

        # Look at recent entertainment detections
        recent_entertainment = [
            r
            for r in self.recent_results[-5:]  # Last 5 results
            if r.is_entertainment and r.overall_confidence > 0.5
        ]

        if recent_entertainment:
            # Boost confidence if recent entertainment detected
            persistence_boost = min(len(recent_entertainment) * 0.05, 0.15)
            confidence += persistence_boost

            self.logger.debug(f"Temporal boost applied: +{persistence_boost:.3f}")

        return confidence

    def _determine_action(
        self, confidence: float, category: ContentCategory
    ) -> ActionType:
        """Determine the recommended action based on confidence and category"""
        # No action for productive content
        if category == ContentCategory.PRODUCTIVE:
            return ActionType.LOG_ONLY

        # Conservative approach for unknown content
        if category == ContentCategory.UNKNOWN:
            if confidence > 0.8:
                return ActionType.WARN
            else:
                return ActionType.LOG_ONLY

        # Action based on confidence thresholds for entertainment content
        if confidence >= self.block_threshold:
            return ActionType.BLOCK
        elif confidence >= self.warn_threshold:
            return ActionType.WARN
        elif confidence >= self.log_threshold:
            return ActionType.LOG_ONLY
        else:
            return ActionType.ALLOW

    def _update_history(self, result: AggregatedResult) -> None:
        """Update the history of aggregated results"""
        self.recent_results.append(result)

        # Keep only recent results
        if len(self.recent_results) > self.max_history_size:
            self.recent_results = self.recent_results[-self.max_history_size :]

    def _create_no_detection_result(self, start_time: datetime) -> AggregatedResult:
        """Create result when no valid detections are available"""
        duration = int((datetime.now() - start_time).total_seconds() * 1000)

        return AggregatedResult(
            overall_confidence=0.0,
            primary_category=ContentCategory.UNKNOWN,
            recommended_action=ActionType.ALLOW,
            individual_results=[],
            timestamp=datetime.now(),
            analysis_duration_ms=duration,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregation statistics for monitoring"""
        if not self.recent_results:
            return {"recent_results": 0}

        recent_confidences = [r.overall_confidence for r in self.recent_results]
        entertainment_ratio = sum(
            1 for r in self.recent_results if r.is_entertainment
        ) / len(self.recent_results)

        return {
            "recent_results": len(self.recent_results),
            "avg_confidence": mean(recent_confidences),
            "median_confidence": median(recent_confidences),
            "entertainment_ratio": entertainment_ratio,
            "recent_categories": [
                r.primary_category.value for r in self.recent_results[-5:]
            ],
            "recent_actions": [
                r.recommended_action.value for r in self.recent_results[-5:]
            ],
        }
