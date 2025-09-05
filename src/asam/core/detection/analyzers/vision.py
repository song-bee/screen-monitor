"""
Computer Vision Analyzer

Analyzes screen captures for entertainment content detection using computer vision.
"""

import io
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from ..types import AnalysisType, ContentCategory, DetectionResult, ScreenCapture
from .base import AnalyzerBase


class VisionAnalyzer(AnalyzerBase):
    """Analyzes screen captures for entertainment content using computer vision"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.min_image_size = self.config.get("min_image_size", (100, 100))
        self.enable_ad_detection = self.config.get("enable_ad_detection", True)
        self.enable_video_detection = self.config.get("enable_video_detection", True)
        self.enable_game_detection = self.config.get("enable_game_detection", True)

        # Detection thresholds
        self.video_motion_threshold = self.config.get("video_motion_threshold", 0.3)
        self.ad_template_threshold = self.config.get("ad_template_threshold", 0.8)
        self.game_ui_threshold = self.config.get("game_ui_threshold", 0.6)

        # Store previous frame for motion detection
        self.previous_frame: Optional[np.ndarray] = None

    @property
    def analyzer_type(self) -> AnalysisType:
        return AnalysisType.VISION

    async def analyze(self, data: ScreenCapture) -> Optional[DetectionResult]:
        """
        Analyze screen capture for entertainment content

        Args:
            data: ScreenCapture object with image data

        Returns:
            DetectionResult with confidence and category
        """
        if not self.should_analyze(data):
            return None

        try:
            # Convert image data to OpenCV format
            image = self._load_image(data.image_data)
            if image is None:
                return None

            # Perform various detection analyses
            evidence = {}
            confidence_scores = []
            detected_categories = []

            # Video content detection (motion analysis)
            if self.enable_video_detection:
                video_result = await self._detect_video_content(image)
                if video_result:
                    evidence.update(video_result["evidence"])
                    confidence_scores.append(video_result["confidence"])
                    detected_categories.append(ContentCategory.VIDEO_STREAMING)

            # Advertisement detection
            if self.enable_ad_detection:
                ad_result = await self._detect_advertisements(image)
                if ad_result:
                    evidence.update(ad_result["evidence"])
                    confidence_scores.append(ad_result["confidence"])
                    detected_categories.append(ContentCategory.ENTERTAINMENT)

            # Gaming UI detection
            if self.enable_game_detection:
                game_result = await self._detect_gaming_ui(image)
                if game_result:
                    evidence.update(game_result["evidence"])
                    confidence_scores.append(game_result["confidence"])
                    detected_categories.append(ContentCategory.GAMING)

            # Color richness analysis (entertainment often more colorful)
            color_result = await self._analyze_color_richness(image)
            if color_result:
                evidence.update(color_result["evidence"])
                if color_result["confidence"] > 0.3:  # Lower threshold for color
                    confidence_scores.append(
                        color_result["confidence"] * 0.3
                    )  # Weighted down

            # Store current frame for next analysis
            self.previous_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate overall confidence and category
            if not confidence_scores:
                return self._create_low_confidence_result(evidence)

            overall_confidence = max(confidence_scores)
            primary_category = self._determine_primary_category(
                detected_categories, confidence_scores
            )

            # Add technical evidence
            evidence.update(
                {
                    "image_resolution": data.screen_resolution,
                    "active_window": data.active_window_title,
                    "active_process": data.active_process_name,
                    "detection_methods": list(evidence.keys()),
                    "confidence_breakdown": dict(
                        zip(
                            [cat.value for cat in detected_categories],
                            confidence_scores,
                        )
                    ),
                }
            )

            return DetectionResult(
                analyzer_type=self.analyzer_type,
                confidence=overall_confidence,
                category=primary_category,
                evidence=evidence,
                timestamp=datetime.now(),
                metadata={
                    "image_size": image.shape[:2],
                    "detection_count": len(confidence_scores),
                },
            )

        except Exception as e:
            self.logger.error(f"Error in vision analysis: {e}")
            return None

    def should_analyze(self, data: ScreenCapture) -> bool:
        """Check if screen capture should be analyzed"""
        if not super().should_analyze(data):
            return False

        # Check image size
        if (
            data.screen_resolution[0] < self.min_image_size[0]
            or data.screen_resolution[1] < self.min_image_size[1]
        ):
            return False

        return True

    def _load_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Load image from bytes into OpenCV format"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert to OpenCV format (BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            return opencv_image
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None

    async def _detect_video_content(
        self, image: np.ndarray
    ) -> Optional[dict[str, Any]]:
        """Detect video content through motion analysis"""
        try:
            current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.previous_frame is None:
                return None

            # Calculate frame difference
            frame_diff = cv2.absdiff(current_gray, self.previous_frame)

            # Calculate motion score
            motion_pixels = np.sum(frame_diff > 30)  # Pixels with significant change
            total_pixels = frame_diff.shape[0] * frame_diff.shape[1]
            motion_ratio = motion_pixels / total_pixels

            if motion_ratio > self.video_motion_threshold:
                # Additional checks for video-like motion patterns
                motion_magnitude = np.mean(frame_diff[frame_diff > 30])

                return {
                    "confidence": min(motion_ratio * 2, 1.0),  # Scale motion ratio
                    "evidence": {
                        "motion_ratio": float(motion_ratio),
                        "motion_magnitude": float(motion_magnitude),
                        "motion_pixels": int(motion_pixels),
                        "detection_type": "video_motion",
                    },
                }

            return None
        except Exception as e:
            self.logger.error(f"Error in video detection: {e}")
            return None

    async def _detect_advertisements(
        self, image: np.ndarray
    ) -> Optional[dict[str, Any]]:
        """Detect advertisement patterns"""
        try:
            # Common ad detection patterns
            evidence = {"detection_type": "advertisement"}
            confidence_factors = []

            # Detect common ad aspect ratios (banners, rectangles)
            height, width = image.shape[:2]
            aspect_ratio = width / height

            # Banner-like ratios (very wide)
            if aspect_ratio > 6 or aspect_ratio < 0.2:
                confidence_factors.append(0.4)
                evidence["banner_aspect_ratio"] = float(aspect_ratio)

            # Detect bright, saturated colors (common in ads)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            high_saturation_ratio = np.sum(saturation > 200) / saturation.size

            if high_saturation_ratio > 0.3:
                confidence_factors.append(high_saturation_ratio)
                evidence["high_saturation_ratio"] = float(high_saturation_ratio)

            # Detect text regions (ads often have prominent text)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Count rectangular text-like regions
            rect_regions = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 10 and w / h > 2:  # Text-like rectangles
                    rect_regions += 1

            if rect_regions > 3:
                text_confidence = min(rect_regions / 10, 0.5)
                confidence_factors.append(text_confidence)
                evidence["text_regions"] = rect_regions

            if confidence_factors:
                overall_confidence = max(confidence_factors)
                evidence["confidence_factors"] = confidence_factors
                return {"confidence": overall_confidence, "evidence": evidence}

            return None
        except Exception as e:
            self.logger.error(f"Error in ad detection: {e}")
            return None

    async def _detect_gaming_ui(self, image: np.ndarray) -> Optional[dict[str, Any]]:
        """Detect gaming UI elements"""
        try:
            evidence = {"detection_type": "gaming_ui"}
            confidence_factors = []

            # Detect UI elements common in games
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Health/mana bars (horizontal rectangles, often with gradients)

            # Create simple horizontal bar template
            bar_template = np.ones((10, 100), dtype=np.uint8) * 128
            bar_template[2:8, 5:95] = 255  # White bar

            result = cv2.matchTemplate(gray, bar_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.3)
            bar_matches = len(locations[0])

            if bar_matches > 2:
                confidence_factors.append(min(bar_matches / 10, 0.6))
                evidence["ui_bars"] = bar_matches

            # Minimap detection (circular or square regions in corners)
            height, width = image.shape[:2]
            corner_size = 150

            # Check corners for minimap-like patterns
            corners = [
                gray[:corner_size, :corner_size],  # Top-left
                gray[:corner_size, -corner_size:],  # Top-right
                gray[-corner_size:, :corner_size],  # Bottom-left
                gray[-corner_size:, -corner_size:],  # Bottom-right
            ]

            minimap_indicators = 0
            for i, corner in enumerate(corners):
                # Look for distinct regions (high contrast)
                contrast = np.std(corner)
                if contrast > 50:
                    minimap_indicators += 1
                    evidence[f"corner_{i}_contrast"] = float(contrast)

            if minimap_indicators > 0:
                confidence_factors.append(minimap_indicators * 0.2)
                evidence["minimap_indicators"] = minimap_indicators

            # HUD element detection (text overlays, numbers)
            contours, _ = cv2.findContours(
                cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1],
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            small_text_regions = sum(
                1
                for c in contours
                if cv2.boundingRect(c)[2] < 100 and cv2.boundingRect(c)[3] < 30
            )

            if small_text_regions > 10:
                confidence_factors.append(min(small_text_regions / 30, 0.4))
                evidence["hud_text_regions"] = small_text_regions

            if confidence_factors:
                overall_confidence = max(confidence_factors)
                evidence["confidence_factors"] = confidence_factors
                return {"confidence": overall_confidence, "evidence": evidence}

            return None
        except Exception as e:
            self.logger.error(f"Error in gaming UI detection: {e}")
            return None

    async def _analyze_color_richness(self, image: np.ndarray) -> dict[str, Any]:
        """Analyze color richness - entertainment content often more colorful"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Calculate color diversity metrics
            hue_histogram = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            saturation_mean = np.mean(hsv[:, :, 1])
            value_mean = np.mean(hsv[:, :, 2])

            # Color diversity (how many different hues are present)
            non_zero_hues = np.sum(hue_histogram > 0)
            hue_diversity = non_zero_hues / 180

            # Saturation richness
            high_saturation_ratio = np.sum(hsv[:, :, 1] > 100) / hsv[:, :, 1].size

            # Calculate entertainment likelihood based on color properties
            color_entertainment_score = (
                hue_diversity * 0.4
                + high_saturation_ratio * 0.4
                + min(saturation_mean / 255, 1.0) * 0.2
            )

            evidence = {
                "detection_type": "color_analysis",
                "hue_diversity": float(hue_diversity),
                "saturation_mean": float(saturation_mean),
                "value_mean": float(value_mean),
                "high_saturation_ratio": float(high_saturation_ratio),
                "color_entertainment_score": float(color_entertainment_score),
            }

            return {"confidence": color_entertainment_score, "evidence": evidence}

        except Exception as e:
            self.logger.error(f"Error in color analysis: {e}")
            return {
                "confidence": 0.0,
                "evidence": {"detection_type": "color_analysis", "error": str(e)},
            }

    def _determine_primary_category(
        self, categories: list[ContentCategory], scores: list[float]
    ) -> ContentCategory:
        """Determine the primary content category from detections"""
        if not categories:
            return ContentCategory.UNKNOWN

        # Weight categories by their confidence scores
        category_scores = {}
        for cat, score in zip(categories, scores):
            if cat in category_scores:
                category_scores[cat] = max(category_scores[cat], score)
            else:
                category_scores[cat] = score

        # Return category with highest score
        return max(category_scores.keys(), key=lambda x: category_scores[x])

    def _create_low_confidence_result(
        self, evidence: dict[str, Any]
    ) -> DetectionResult:
        """Create a low-confidence result when no clear detection is made"""
        return DetectionResult(
            analyzer_type=self.analyzer_type,
            confidence=0.1,
            category=ContentCategory.UNKNOWN,
            evidence=evidence or {"detection_type": "no_detection"},
            timestamp=datetime.now(),
            metadata={"low_confidence": True},
        )
