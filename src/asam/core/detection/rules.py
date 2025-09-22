"""
Advanced Detection Rules Engine

Implements intelligent rules for aggregating multi-analyzer results and making smart decisions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import ActionType, AnalysisType, ContentCategory, DetectionResult


class RuleCondition(Enum):
    """Types of rule conditions"""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CATEGORY_MATCH = "category_match"
    ANALYZER_AGREEMENT = "analyzer_agreement"
    TIME_BASED = "time_based"
    PATTERN_MATCH = "pattern_match"


@dataclass
class DetectionRule:
    """A single detection rule"""

    name: str
    conditions: List[Dict[str, Any]]
    action: ActionType
    priority: int = 0  # Higher priority rules take precedence
    enabled: bool = True

    def evaluate(
        self, results: List[DetectionResult], context: Dict[str, Any] = None
    ) -> bool:
        """Evaluate if this rule matches the current detection results"""
        for condition in self.conditions:
            if not self._evaluate_condition(condition, results, context or {}):
                return False
        return True

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        results: List[DetectionResult],
        context: Dict[str, Any],
    ) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get("type")

        if condition_type == RuleCondition.CONFIDENCE_THRESHOLD.value:
            return self._check_confidence_threshold(condition, results)
        elif condition_type == RuleCondition.CATEGORY_MATCH.value:
            return self._check_category_match(condition, results)
        elif condition_type == RuleCondition.ANALYZER_AGREEMENT.value:
            return self._check_analyzer_agreement(condition, results)
        elif condition_type == RuleCondition.TIME_BASED.value:
            return self._check_time_based(condition, context)
        elif condition_type == RuleCondition.PATTERN_MATCH.value:
            return self._check_pattern_match(condition, results)

        return False

    def _check_confidence_threshold(
        self, condition: Dict[str, Any], results: List[DetectionResult]
    ) -> bool:
        """Check if any analyzer meets confidence threshold"""
        threshold = condition.get("threshold", 0.7)
        analyzer_type = condition.get("analyzer_type")

        for result in results:
            if analyzer_type and result.analyzer_type.value != analyzer_type:
                continue
            if result.confidence >= threshold:
                return True
        return False

    def _check_category_match(
        self, condition: Dict[str, Any], results: List[DetectionResult]
    ) -> bool:
        """Check if any result matches specified category"""
        target_category = condition.get("category")
        min_confidence = condition.get("min_confidence", 0.5)

        for result in results:
            if (
                result.category.value == target_category
                and result.confidence >= min_confidence
            ):
                return True
        return False

    def _check_analyzer_agreement(
        self, condition: Dict[str, Any], results: List[DetectionResult]
    ) -> bool:
        """Check if multiple analyzers agree on classification"""
        min_analyzers = condition.get("min_analyzers", 2)
        category = condition.get("category")
        min_confidence = condition.get("min_confidence", 0.5)

        agreeing_analyzers = 0
        for result in results:
            if (
                result.category.value == category
                and result.confidence >= min_confidence
            ):
                agreeing_analyzers += 1

        return agreeing_analyzers >= min_analyzers

    def _check_time_based(
        self, condition: Dict[str, Any], context: Dict[str, Any]
    ) -> bool:
        """Check time-based conditions"""
        time_type = condition.get("time_type")

        if time_type == "consecutive_detections":
            min_count = condition.get("min_count", 3)
            consecutive_count = context.get("consecutive_entertainment_count", 0)
            return consecutive_count >= min_count
        elif time_type == "time_of_day":
            current_hour = datetime.now().hour
            start_hour = condition.get("start_hour", 0)
            end_hour = condition.get("end_hour", 23)
            return start_hour <= current_hour <= end_hour

        return False

    def _check_pattern_match(
        self, condition: Dict[str, Any], results: List[DetectionResult]
    ) -> bool:
        """Check for specific evidence patterns"""
        pattern_type = condition.get("pattern_type")

        if pattern_type == "gaming_ui_detected":
            for result in results:
                if result.analyzer_type == AnalysisType.VISION and "gaming_ui" in str(
                    result.evidence
                ):
                    return True
        elif pattern_type == "video_streaming":
            for result in results:
                if (
                    result.analyzer_type == AnalysisType.VISION
                    and result.evidence.get("detection_type") == "video_motion"
                ):
                    return True
        elif pattern_type == "social_media_keywords":
            for result in results:
                if result.analyzer_type == AnalysisType.TEXT:
                    keywords = result.evidence.get("keywords", [])
                    social_keywords = [
                        "facebook",
                        "twitter",
                        "instagram",
                        "tiktok",
                        "reddit",
                        "social",
                    ]
                    if any(keyword.lower() in social_keywords for keyword in keywords):
                        return True

        return False


class AdvancedDetectionRulesEngine:
    """Advanced rules engine for intelligent detection decisions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rules: List[DetectionRule] = []
        self.detection_history: List[Dict[str, Any]] = []
        self.consecutive_entertainment_count = 0
        self.last_entertainment_time: Optional[datetime] = None

        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default detection rules"""

        # Rule 1: High-confidence text detection for entertainment
        self.rules.append(
            DetectionRule(
                name="high_confidence_text_entertainment",
                conditions=[
                    {
                        "type": RuleCondition.CONFIDENCE_THRESHOLD.value,
                        "analyzer_type": "text",
                        "threshold": 0.8,
                    },
                    {
                        "type": RuleCondition.CATEGORY_MATCH.value,
                        "category": "entertainment",
                        "min_confidence": 0.8,
                    },
                ],
                action=ActionType.BLOCK,
                priority=10,
            )
        )

        # Rule 2: Multiple analyzers agree on entertainment
        self.rules.append(
            DetectionRule(
                name="multi_analyzer_entertainment_agreement",
                conditions=[
                    {
                        "type": RuleCondition.ANALYZER_AGREEMENT.value,
                        "category": "entertainment",
                        "min_analyzers": 2,
                        "min_confidence": 0.6,
                    }
                ],
                action=ActionType.WARN,
                priority=8,
            )
        )

        # Rule 3: Gaming UI detected with high confidence
        self.rules.append(
            DetectionRule(
                name="gaming_ui_detected",
                conditions=[
                    {
                        "type": RuleCondition.CONFIDENCE_THRESHOLD.value,
                        "analyzer_type": "vision",
                        "threshold": 0.7,
                    },
                    {
                        "type": RuleCondition.PATTERN_MATCH.value,
                        "pattern_type": "gaming_ui_detected",
                    },
                ],
                action=ActionType.BLOCK,
                priority=9,
            )
        )

        # Rule 4: Video streaming detection
        self.rules.append(
            DetectionRule(
                name="video_streaming_detected",
                conditions=[
                    {
                        "type": RuleCondition.CONFIDENCE_THRESHOLD.value,
                        "analyzer_type": "vision",
                        "threshold": 0.6,
                    },
                    {
                        "type": RuleCondition.PATTERN_MATCH.value,
                        "pattern_type": "video_streaming",
                    },
                ],
                action=ActionType.WARN,
                priority=7,
            )
        )

        # Rule 5: Consecutive entertainment detections
        self.rules.append(
            DetectionRule(
                name="consecutive_entertainment_detections",
                conditions=[
                    {
                        "type": RuleCondition.TIME_BASED.value,
                        "time_type": "consecutive_detections",
                        "min_count": 3,
                    }
                ],
                action=ActionType.BLOCK,
                priority=6,
            )
        )

        # Rule 6: Social media detected
        self.rules.append(
            DetectionRule(
                name="social_media_detected",
                conditions=[
                    {
                        "type": RuleCondition.PATTERN_MATCH.value,
                        "pattern_type": "social_media_keywords",
                    },
                    {
                        "type": RuleCondition.CONFIDENCE_THRESHOLD.value,
                        "analyzer_type": "text",
                        "threshold": 0.5,
                    },
                ],
                action=ActionType.WARN,
                priority=5,
            )
        )

        # Rule 7: Work hours protection (stricter during work hours)
        self.rules.append(
            DetectionRule(
                name="work_hours_entertainment",
                conditions=[
                    {
                        "type": RuleCondition.TIME_BASED.value,
                        "time_type": "time_of_day",
                        "start_hour": 9,
                        "end_hour": 17,
                    },
                    {
                        "type": RuleCondition.CATEGORY_MATCH.value,
                        "category": "entertainment",
                        "min_confidence": 0.5,
                    },
                ],
                action=ActionType.BLOCK,
                priority=8,
            )
        )

        # Rule 8: Default allow for productive content
        self.rules.append(
            DetectionRule(
                name="productive_content_allow",
                conditions=[
                    {
                        "type": RuleCondition.CATEGORY_MATCH.value,
                        "category": "productive",
                        "min_confidence": 0.6,
                    }
                ],
                action=ActionType.ALLOW,
                priority=3,
            )
        )

    def evaluate_detection_results(
        self, results: List[DetectionResult], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate detection results using advanced rules

        Returns:
            Dict containing recommended action, confidence, reasoning, etc.
        """
        if not results:
            return self._create_default_result()

        # Update consecutive detection tracking
        self._update_consecutive_tracking(results)

        # Build context for rule evaluation
        evaluation_context = {
            "consecutive_entertainment_count": self.consecutive_entertainment_count,
            "last_entertainment_time": self.last_entertainment_time,
            **(context or {}),
        }

        # Evaluate all rules and find matches
        matched_rules = []
        for rule in self.rules:
            if rule.enabled and rule.evaluate(results, evaluation_context):
                matched_rules.append(rule)

        # Sort by priority (highest first)
        matched_rules.sort(key=lambda r: r.priority, reverse=True)

        # Determine final action and confidence
        if matched_rules:
            primary_rule = matched_rules[0]
            final_action = primary_rule.action

            # Calculate confidence based on rule matches and analyzer confidences
            confidence = self._calculate_final_confidence(results, matched_rules)

            # Determine primary category
            category = self._determine_primary_category(results, matched_rules)

            reasoning = self._build_reasoning(matched_rules, results)
        else:
            # No rules matched - use fallback logic
            final_action = ActionType.LOG_ONLY
            confidence = max([r.confidence for r in results]) if results else 0.1
            category = self._get_highest_confidence_category(results)
            reasoning = "No specific rules matched, using default behavior"

        # Store in history
        self._add_to_history(results, final_action, confidence, category)

        return {
            "action": final_action,
            "confidence": confidence,
            "category": category,
            "reasoning": reasoning,
            "matched_rules": [r.name for r in matched_rules],
            "analyzer_breakdown": self._get_analyzer_breakdown(results),
            "consecutive_count": self.consecutive_entertainment_count,
        }

    def _update_consecutive_tracking(self, results: List[DetectionResult]) -> None:
        """Update consecutive entertainment detection tracking"""
        has_entertainment = any(
            r.category
            in [
                ContentCategory.ENTERTAINMENT,
                ContentCategory.GAMING,
                ContentCategory.VIDEO_STREAMING,
                ContentCategory.SOCIAL_MEDIA,
            ]
            and r.confidence > 0.5
            for r in results
        )

        if has_entertainment:
            self.consecutive_entertainment_count += 1
            self.last_entertainment_time = datetime.now()
        else:
            # Reset if no entertainment detected or if too much time passed
            if (
                not self.last_entertainment_time
                or datetime.now() - self.last_entertainment_time > timedelta(minutes=5)
            ):
                self.consecutive_entertainment_count = 0

    def _calculate_final_confidence(
        self, results: List[DetectionResult], matched_rules: List[DetectionRule]
    ) -> float:
        """Calculate final confidence score"""
        if not results:
            return 0.1

        # Base confidence from highest analyzer
        base_confidence = max(r.confidence for r in results)

        # Boost confidence if multiple rules match
        rule_boost = min(len(matched_rules) * 0.1, 0.3)

        # Boost if multiple analyzers agree
        entertainment_results = [
            r
            for r in results
            if r.category
            in [
                ContentCategory.ENTERTAINMENT,
                ContentCategory.GAMING,
                ContentCategory.VIDEO_STREAMING,
                ContentCategory.SOCIAL_MEDIA,
            ]
            and r.confidence > 0.4
        ]

        agreement_boost = min(len(entertainment_results) * 0.15, 0.4)

        final_confidence = min(base_confidence + rule_boost + agreement_boost, 1.0)
        return round(final_confidence, 3)

    def _determine_primary_category(
        self, results: List[DetectionResult], matched_rules: List[DetectionRule]
    ) -> ContentCategory:
        """Determine the primary content category"""
        if not results:
            return ContentCategory.UNKNOWN

        # Count categories by confidence-weighted votes
        category_scores = {}
        for result in results:
            category = result.category
            if category not in category_scores:
                category_scores[category] = 0
            category_scores[category] += result.confidence

        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])

        return ContentCategory.UNKNOWN

    def _get_highest_confidence_category(
        self, results: List[DetectionResult]
    ) -> ContentCategory:
        """Get category from highest confidence result"""
        if not results:
            return ContentCategory.UNKNOWN

        highest_result = max(results, key=lambda r: r.confidence)
        return highest_result.category

    def _build_reasoning(
        self, matched_rules: List[DetectionRule], results: List[DetectionResult]
    ) -> str:
        """Build human-readable reasoning for the decision"""
        if not matched_rules:
            return "No specific rules triggered"

        primary_rule = matched_rules[0]
        analyzer_info = []

        for result in results:
            if result.confidence > 0.3:
                analyzer_info.append(
                    f"{result.analyzer_type.value}({result.confidence:.2f})"
                )

        reasoning_parts = [
            f"Primary rule: {primary_rule.name}",
            f"Analyzers: {', '.join(analyzer_info)}",
        ]

        if len(matched_rules) > 1:
            reasoning_parts.append(
                f"Additional rules: {', '.join(r.name for r in matched_rules[1:3])}"
            )

        if self.consecutive_entertainment_count > 1:
            reasoning_parts.append(
                f"Consecutive detections: {self.consecutive_entertainment_count}"
            )

        return " | ".join(reasoning_parts)

    def _get_analyzer_breakdown(
        self, results: List[DetectionResult]
    ) -> Dict[str, float]:
        """Get breakdown of analyzer confidences"""
        breakdown = {}
        for result in results:
            breakdown[result.analyzer_type.value] = round(result.confidence, 3)
        return breakdown

    def _add_to_history(
        self,
        results: List[DetectionResult],
        action: ActionType,
        confidence: float,
        category: ContentCategory,
    ) -> None:
        """Add detection to history for pattern analysis"""
        history_entry = {
            "timestamp": datetime.now(),
            "action": action,
            "confidence": confidence,
            "category": category,
            "analyzer_count": len(results),
            "max_analyzer_confidence": (
                max(r.confidence for r in results) if results else 0
            ),
        }

        self.detection_history.append(history_entry)

        # Keep only recent history (last 100 entries)
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result when no detection results available"""
        return {
            "action": ActionType.LOG_ONLY,
            "confidence": 0.1,
            "category": ContentCategory.UNKNOWN,
            "reasoning": "No detection results available",
            "matched_rules": [],
            "analyzer_breakdown": {},
            "consecutive_count": 0,
        }

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule performance"""
        total_evaluations = len(self.detection_history)
        if total_evaluations == 0:
            return {"total_evaluations": 0}

        # Calculate action distribution
        action_counts = {}
        category_counts = {}

        for entry in self.detection_history:
            action = entry["action"]
            category = entry["category"]

            action_counts[action.value] = action_counts.get(action.value, 0) + 1
            category_counts[category.value] = category_counts.get(category.value, 0) + 1

        return {
            "total_evaluations": total_evaluations,
            "action_distribution": action_counts,
            "category_distribution": category_counts,
            "consecutive_entertainment_streak": self.consecutive_entertainment_count,
            "avg_confidence": sum(e["confidence"] for e in self.detection_history)
            / total_evaluations,
        }

    def add_custom_rule(self, rule: DetectionRule) -> None:
        """Add a custom detection rule"""
        self.rules.append(rule)
        # Re-sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule by name"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                return True
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule by name"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                return True
        return False
