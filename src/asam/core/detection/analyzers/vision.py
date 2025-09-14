"""
Computer Vision Analyzer

Analyzes screen captures for entertainment content detection using computer vision.
"""

from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

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
            data: ScreenCapture object with PIL image and numpy array

        Returns:
            DetectionResult with confidence and category
        """
        if not self.should_analyze(data):
            return None

        try:
            # Use the numpy array from ScreenCapture directly
            image = self._convert_to_opencv(data.image_array)
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
                    "image_resolution": (data.width, data.height),
                    "active_window": data.active_window_title,
                    "active_process": data.active_process_name,
                    "capture_source": data.source,
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
        if data.width < self.min_image_size[0] or data.height < self.min_image_size[1]:
            return False

        return True

    def _convert_to_opencv(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Convert numpy array to OpenCV format"""
        try:
            # Ensure we have a 3-channel image
            if len(image_array.shape) == 3:
                # Convert RGB to BGR for OpenCV
                if image_array.shape[2] == 3:
                    opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                elif image_array.shape[2] == 4:
                    # RGBA to BGR
                    opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                else:
                    self.logger.error(f"Unsupported image format: {image_array.shape}")
                    return None
            else:
                self.logger.error(f"Invalid image array shape: {image_array.shape}")
                return None

            return opencv_image
        except Exception as e:
            self.logger.error(f"Error converting image array: {e}")
            return None

    async def _detect_video_content(
        self, image: np.ndarray
    ) -> Optional[dict[str, Any]]:
        """Detect video content through advanced motion analysis"""
        try:
            current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.previous_frame is None:
                return None

            # Multiple motion detection approaches
            evidence = {"detection_type": "video_motion"}
            confidence_factors = []

            # 1. Frame difference analysis
            frame_diff = cv2.absdiff(current_gray, self.previous_frame)
            motion_pixels = np.sum(frame_diff > 30)
            total_pixels = frame_diff.shape[0] * frame_diff.shape[1]
            motion_ratio = motion_pixels / total_pixels

            if motion_ratio > self.video_motion_threshold:
                motion_magnitude = np.mean(frame_diff[frame_diff > 30])
                confidence_factors.append(min(motion_ratio * 2, 1.0))
                evidence.update(
                    {
                        "motion_ratio": float(motion_ratio),
                        "motion_magnitude": float(motion_magnitude),
                        "motion_pixels": int(motion_pixels),
                    }
                )

            # 2. Optical Flow analysis (Lucas-Kanade)
            flow_confidence = await self._analyze_optical_flow(
                current_gray, self.previous_frame
            )
            if flow_confidence > 0.3:
                confidence_factors.append(flow_confidence)
                evidence["optical_flow_confidence"] = float(flow_confidence)

            # 3. Temporal consistency (video has consistent motion patterns)
            if hasattr(self, "_motion_history"):
                consistency_score = self._calculate_motion_consistency(motion_ratio)
                if consistency_score > 0.4:
                    confidence_factors.append(consistency_score * 0.8)  # Weighted
                    evidence["temporal_consistency"] = float(consistency_score)
            else:
                self._motion_history = []

            # Store motion history for temporal analysis
            self._motion_history.append(motion_ratio)
            if len(self._motion_history) > 10:  # Keep last 10 frames
                self._motion_history.pop(0)

            # 4. Find most active motion areas and analyze them
            motion_areas = self._find_active_motion_areas(frame_diff, current_gray)
            if motion_areas:
                area_confidence = await self._analyze_focused_motion_areas(
                    motion_areas, current_gray, self.previous_frame, image
                )
                if area_confidence > 0.4:
                    confidence_factors.append(area_confidence)
                    evidence["focused_motion_confidence"] = float(area_confidence)
                    evidence["motion_areas_count"] = len(motion_areas)
                    
            # 5. Legacy region-based motion analysis (kept for comparison)
            region_confidence = self._analyze_motion_regions(frame_diff)
            if region_confidence > 0.3:
                confidence_factors.append(region_confidence * 0.5)  # Lower weight now
                evidence["region_motion_confidence"] = float(region_confidence)

            if confidence_factors:
                overall_confidence = max(confidence_factors)
                evidence["confidence_factors"] = confidence_factors
                evidence["analysis_methods"] = len(confidence_factors)
                return {"confidence": overall_confidence, "evidence": evidence}

            return None
        except Exception as e:
            self.logger.error(f"Error in video detection: {e}")
            return None

    async def _analyze_optical_flow(
        self, current_gray: np.ndarray, previous_gray: np.ndarray
    ) -> float:
        """Analyze optical flow to detect smooth motion patterns typical of video"""
        try:
            # Use sparse optical flow (Lucas-Kanade)
            # First, detect corners in previous frame
            corners = cv2.goodFeaturesToTrack(
                previous_gray,
                maxCorners=100,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=3,
            )

            if corners is None or len(corners) < 10:
                return 0.0

            # Calculate optical flow
            flow_vectors, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, corners, None
            )

            # Filter good tracks
            good_new = flow_vectors[status == 1]
            good_old = corners[status == 1]

            if len(good_new) < 5:
                return 0.0

            # Calculate flow magnitude and consistency
            flow_magnitudes = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
            avg_magnitude = np.mean(flow_magnitudes)
            magnitude_std = np.std(flow_magnitudes)

            # Video content typically has consistent, moderate motion
            if 2 < avg_magnitude < 20 and magnitude_std < avg_magnitude:
                consistency = 1.0 - (magnitude_std / max(avg_magnitude, 1))
                return min(consistency * (avg_magnitude / 20), 1.0)

            return 0.0
        except Exception as e:
            self.logger.debug(f"Optical flow analysis failed: {e}")
            return 0.0

    def _calculate_motion_consistency(self, current_motion: float) -> float:
        """Calculate temporal consistency of motion (video has steady motion patterns)"""
        if len(self._motion_history) < 3:
            return 0.0

        # Check for consistent motion over time
        recent_motion = self._motion_history[-3:]
        motion_mean = np.mean(recent_motion)
        motion_std = np.std(recent_motion)

        # Video content typically has consistent moderate motion
        if 0.1 < motion_mean < 0.8 and motion_std < 0.3:
            consistency = 1.0 - (motion_std / max(motion_mean, 0.1))
            return min(consistency, 1.0)

        return 0.0

    def _analyze_motion_regions(self, frame_diff: np.ndarray) -> float:
        """Analyze motion distribution - videos often have centered or distributed motion"""
        try:
            height, width = frame_diff.shape

            # Divide image into regions
            center_region = frame_diff[
                height // 4 : 3 * height // 4, width // 4 : 3 * width // 4
            ]
            edge_regions = [
                frame_diff[: height // 4, :],  # Top
                frame_diff[3 * height // 4 :, :],  # Bottom
                frame_diff[:, : width // 4],  # Left
                frame_diff[:, 3 * width // 4 :],  # Right
            ]

            # Calculate motion in each region
            center_motion = np.mean(center_region > 30)
            edge_motion = np.mean([np.mean(region > 30) for region in edge_regions])

            # Videos often have more motion in center, or distributed motion
            if center_motion > edge_motion * 1.5:  # Center-heavy motion
                return min(center_motion * 1.5, 1.0)
            elif center_motion > 0.1 and edge_motion > 0.05:  # Distributed motion
                return min((center_motion + edge_motion) * 0.8, 1.0)

            return 0.0
        except Exception:
            return 0.0

    def _find_active_motion_areas(
        self, frame_diff: np.ndarray, current_gray: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Find rectangular areas with the most motion activity"""
        try:
            height, width = frame_diff.shape
            motion_threshold = 30
            
            # Create binary motion mask
            motion_mask = (frame_diff > motion_threshold).astype(np.uint8) * 255
            
            # Use morphological operations to connect nearby motion areas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel // 2)
            
            # Find contours of motion areas
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_areas = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter based on size and shape
                if (
                    area > 5000 and  # Minimum area (roughly 70x70 pixels)
                    area < width * height * 0.8 and  # Not too large (max 80% of screen)
                    w > 60 and h > 60 and  # Minimum dimensions
                    min(w, h) / max(w, h) > 0.3  # Not too thin/narrow
                ):
                    # Calculate motion intensity in this area
                    roi_motion = frame_diff[y:y+h, x:x+w]
                    motion_intensity = np.mean(roi_motion[roi_motion > motion_threshold])
                    motion_pixel_ratio = np.sum(roi_motion > motion_threshold) / (w * h)
                    
                    # Only include areas with significant motion
                    if motion_intensity > 40 and motion_pixel_ratio > 0.1:
                        motion_areas.append((x, y, w, h, motion_intensity, motion_pixel_ratio))
            
            # Sort by motion intensity and keep top areas
            motion_areas.sort(key=lambda area: area[4] * area[5], reverse=True)  # intensity * ratio
            
            # Return top 3 motion areas (without intensity metrics for external use)
            return [(x, y, w, h) for x, y, w, h, _, _ in motion_areas[:3]]
            
        except Exception as e:
            self.logger.debug(f"Error finding motion areas: {e}")
            return []

    async def _analyze_focused_motion_areas(
        self, 
        motion_areas: list[tuple[int, int, int, int]], 
        current_gray: np.ndarray, 
        previous_gray: np.ndarray,
        color_image: np.ndarray
    ) -> float:
        """Analyze focused motion areas to determine if they contain video/entertainment content"""
        try:
            area_confidences = []
            
            for x, y, w, h in motion_areas:
                # Extract the motion area from both frames
                current_roi = current_gray[y:y+h, x:x+w]
                previous_roi = previous_gray[y:y+h, x:x+w]
                color_roi = color_image[y:y+h, x:x+w]
                
                area_confidence = 0.0
                evidence_factors = []
                
                # 1. Temporal consistency analysis in focused area
                roi_diff = cv2.absdiff(current_roi, previous_roi)
                motion_pixels = np.sum(roi_diff > 30)
                total_pixels = current_roi.size
                local_motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
                
                if 0.1 < local_motion_ratio < 0.9:  # Good motion range for video
                    consistency_score = self._calculate_area_motion_consistency(
                        local_motion_ratio, x, y, w, h
                    )
                    if consistency_score > 0.3:
                        evidence_factors.append(consistency_score)
                
                # 2. Check for video-like motion patterns (smooth, consistent)
                flow_quality = await self._analyze_area_optical_flow(current_roi, previous_roi)
                if flow_quality > 0.3:
                    evidence_factors.append(flow_quality * 0.9)  # High weight for optical flow
                
                # 3. Color dynamics analysis (videos have changing colors)
                color_dynamics = self._analyze_area_color_dynamics(color_roi, x, y, w, h)
                if color_dynamics > 0.3:
                    evidence_factors.append(color_dynamics * 0.7)
                
                # 4. Check for video-specific characteristics
                video_characteristics = self._detect_video_characteristics_in_area(
                    current_roi, color_roi, w, h
                )
                if video_characteristics > 0.3:
                    evidence_factors.append(video_characteristics * 0.8)
                
                # 5. Aspect ratio analysis (videos often have specific ratios)
                aspect_ratio = w / h if h > 0 else 0
                aspect_score = 0.0
                if 1.3 < aspect_ratio < 2.4:  # Common video aspect ratios (4:3 to 21:9)
                    aspect_score = min(0.6, 1.0 - abs(aspect_ratio - 1.78) / 1.78)  # 16:9 is ideal
                    if aspect_score > 0.2:
                        evidence_factors.append(aspect_score * 0.5)
                
                # Calculate confidence for this area
                if evidence_factors:
                    area_confidence = max(evidence_factors)
                    area_confidences.append(area_confidence)
                    
                    self.logger.debug(
                        f"Motion area ({x},{y},{w}x{h}): "
                        f"motion={local_motion_ratio:.3f}, "
                        f"flow={flow_quality:.3f}, "
                        f"color={color_dynamics:.3f}, "
                        f"video_char={video_characteristics:.3f}, "
                        f"aspect={aspect_score:.3f}, "
                        f"confidence={area_confidence:.3f}"
                    )
            
            # Return the highest confidence from all areas
            return max(area_confidences) if area_confidences else 0.0
            
        except Exception as e:
            self.logger.error(f"Error analyzing focused motion areas: {e}")
            return 0.0

    def _calculate_area_motion_consistency(
        self, motion_ratio: float, x: int, y: int, w: int, h: int
    ) -> float:
        """Calculate motion consistency for a specific area"""
        try:
            # Store per-area motion history
            area_key = f"{x}_{y}_{w}_{h}"
            if not hasattr(self, '_area_motion_history'):
                self._area_motion_history = {}
            
            if area_key not in self._area_motion_history:
                self._area_motion_history[area_key] = []
            
            history = self._area_motion_history[area_key]
            history.append(motion_ratio)
            
            # Keep only recent history
            if len(history) > 5:
                history.pop(0)
            
            if len(history) < 3:
                return 0.0
            
            # Calculate consistency
            mean_motion = np.mean(history)
            std_motion = np.std(history)
            
            # Video content has consistent moderate motion
            if 0.15 < mean_motion < 0.7 and std_motion < 0.2:
                consistency = 1.0 - (std_motion / max(mean_motion, 0.1))
                return min(consistency, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _analyze_area_optical_flow(
        self, current_roi: np.ndarray, previous_roi: np.ndarray
    ) -> float:
        """Analyze optical flow specifically within a motion area"""
        try:
            if current_roi.shape[0] < 50 or current_roi.shape[1] < 50:
                return 0.0  # Too small for meaningful optical flow
            
            # Detect corners in the ROI
            corners = cv2.goodFeaturesToTrack(
                previous_roi,
                maxCorners=50,  # Fewer corners for smaller area
                qualityLevel=0.01,
                minDistance=5,  # Smaller distance for ROI
                blockSize=3
            )
            
            if corners is None or len(corners) < 5:
                return 0.0
            
            # Calculate optical flow
            flow_vectors, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_roi, current_roi, corners, None
            )
            
            # Filter good tracks
            good_new = flow_vectors[status == 1]
            good_old = corners[status == 1]
            
            if len(good_new) < 3:
                return 0.0
            
            # Analyze flow patterns
            flow_magnitudes = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
            avg_magnitude = np.mean(flow_magnitudes)
            magnitude_std = np.std(flow_magnitudes)
            
            # Check for directional consistency (videos often have coherent motion)
            if len(good_new) >= 5:
                flow_vectors_2d = good_new - good_old
                flow_angles = np.arctan2(flow_vectors_2d[:, 1], flow_vectors_2d[:, 0])
                angle_consistency = 1.0 - (np.std(flow_angles) / np.pi)  # Normalize by pi
                
                if (
                    2 < avg_magnitude < 15 and  # Reasonable magnitude for video
                    magnitude_std < avg_magnitude * 0.8 and  # Consistent magnitude
                    angle_consistency > 0.3  # Some directional consistency
                ):
                    quality = min(
                        (avg_magnitude / 15) * angle_consistency * 
                        (1 - magnitude_std / max(avg_magnitude, 1)), 
                        1.0
                    )
                    return quality
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Area optical flow analysis failed: {e}")
            return 0.0

    def _analyze_area_color_dynamics(
        self, color_roi: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        """Analyze color changes in a focused area over time"""
        try:
            # Store per-area color history
            area_key = f"{x}_{y}_{w}_{h}"
            if not hasattr(self, '_area_color_history'):
                self._area_color_history = {}
            
            if area_key not in self._area_color_history:
                self._area_color_history[area_key] = []
            
            # Calculate current color statistics
            hsv_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            color_stats = {
                'mean_hue': np.mean(hsv_roi[:, :, 0]),
                'mean_saturation': np.mean(hsv_roi[:, :, 1]),
                'mean_value': np.mean(hsv_roi[:, :, 2]),
                'hue_std': np.std(hsv_roi[:, :, 0])
            }
            
            history = self._area_color_history[area_key]
            history.append(color_stats)
            
            if len(history) > 4:
                history.pop(0)
            
            if len(history) < 3:
                return 0.0
            
            # Analyze color dynamics
            hue_changes = [abs(history[i]['mean_hue'] - history[i-1]['mean_hue']) 
                          for i in range(1, len(history))]
            sat_changes = [abs(history[i]['mean_saturation'] - history[i-1]['mean_saturation']) 
                          for i in range(1, len(history))]
            val_changes = [abs(history[i]['mean_value'] - history[i-1]['mean_value']) 
                          for i in range(1, len(history))]
            
            avg_hue_change = np.mean(hue_changes)
            avg_sat_change = np.mean(sat_changes)
            avg_val_change = np.mean(val_changes)
            
            # Videos typically have moderate color changes
            if (
                5 < avg_hue_change < 50 and  # Moderate hue changes
                10 < avg_sat_change < 80 and  # Moderate saturation changes  
                5 < avg_val_change < 60  # Moderate brightness changes
            ):
                # Normalize and combine the changes
                dynamics_score = (
                    min(avg_hue_change / 50, 1.0) * 0.4 +
                    min(avg_sat_change / 80, 1.0) * 0.3 +
                    min(avg_val_change / 60, 1.0) * 0.3
                )
                return dynamics_score
            
            return 0.0
            
        except Exception:
            return 0.0

    def _detect_video_characteristics_in_area(
        self, gray_roi: np.ndarray, color_roi: np.ndarray, w: int, h: int
    ) -> float:
        """Detect video-specific characteristics within a focused area"""
        try:
            confidence_factors = []
            
            # 1. Edge density analysis (videos have varying edge content)
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if 0.05 < edge_density < 0.4:  # Good range for video content
                confidence_factors.append(min(edge_density * 2, 0.7))
            
            # 2. Texture analysis (videos have rich, varying textures)
            if gray_roi.shape[0] > 32 and gray_roi.shape[1] > 32:
                try:
                    # Try scikit-image LBP if available
                    from skimage.feature import local_binary_pattern
                    
                    # LBP analysis
                    lbp = local_binary_pattern(gray_roi, 8, 1, method='uniform')
                    lbp_var = np.var(lbp)
                    
                    if lbp_var > 50:  # Good texture variation
                        confidence_factors.append(min(lbp_var / 200, 0.6))
                        
                except ImportError:
                    # Fallback texture analysis using gradients
                    grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
                    texture_strength = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                    
                    if texture_strength > 20:
                        confidence_factors.append(min(texture_strength / 100, 0.5))
            
            # 3. Color richness in the area
            hsv_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            unique_hues = len(np.unique(hsv_roi[:, :, 0]))
            hue_diversity = unique_hues / 180.0
            high_sat_ratio = np.sum(hsv_roi[:, :, 1] > 100) / hsv_roi[:, :, 1].size
            
            if hue_diversity > 0.3 and high_sat_ratio > 0.2:
                color_richness = (hue_diversity * 0.6 + high_sat_ratio * 0.4)
                confidence_factors.append(min(color_richness, 0.6))
            
            # 4. Contrast variation (videos have dynamic contrast)
            contrast = np.std(gray_roi)
            if 25 < contrast < 120:  # Good contrast range for video
                confidence_factors.append(min(contrast / 120, 0.5))
            
            return max(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            self.logger.debug(f"Video characteristics detection failed: {e}")
            return 0.0

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
        """Detect gaming UI elements with advanced pattern recognition"""
        try:
            evidence = {"detection_type": "gaming_ui"}
            confidence_factors = []
            height, width = image.shape[:2]

            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 1. Health/Mana/Progress bars detection
            bar_confidence = await self._detect_ui_bars(gray)
            if bar_confidence > 0.3:
                confidence_factors.append(bar_confidence)
                evidence["ui_bars_confidence"] = float(bar_confidence)

            # 2. Minimap detection (improved algorithm)
            minimap_confidence = self._detect_minimap_regions(gray, width, height)
            if minimap_confidence > 0.3:
                confidence_factors.append(minimap_confidence * 0.8)
                evidence["minimap_confidence"] = float(minimap_confidence)

            # 3. HUD elements detection (overlays, counters, timers)
            hud_confidence = self._detect_hud_elements(gray, image)
            if hud_confidence > 0.3:
                confidence_factors.append(hud_confidence * 0.7)
                evidence["hud_confidence"] = float(hud_confidence)

            # 4. Game-specific color patterns (vibrant UI, specific color schemes)
            color_confidence = self._detect_gaming_colors(hsv)
            if color_confidence > 0.3:
                confidence_factors.append(color_confidence * 0.5)
                evidence["gaming_colors_confidence"] = float(color_confidence)

            # 5. Crosshair/reticle detection
            crosshair_confidence = self._detect_crosshair(gray, width, height)
            if crosshair_confidence > 0.4:
                confidence_factors.append(crosshair_confidence * 0.9)
                evidence["crosshair_confidence"] = float(crosshair_confidence)

            # 6. Action/ability icons detection
            icon_confidence = self._detect_action_icons(image)
            if icon_confidence > 0.3:
                confidence_factors.append(icon_confidence * 0.6)
                evidence["action_icons_confidence"] = float(icon_confidence)

            if confidence_factors:
                overall_confidence = max(confidence_factors)
                evidence["confidence_factors"] = confidence_factors
                evidence["detection_methods"] = len(confidence_factors)
                return {"confidence": overall_confidence, "evidence": evidence}

            return None
        except Exception as e:
            self.logger.error(f"Error in gaming UI detection: {e}")
            return None

    async def _detect_ui_bars(self, gray: np.ndarray) -> float:
        """Detect horizontal/vertical progress bars common in games"""
        try:
            # Create multiple bar templates for different sizes
            bar_confidences = []

            for bar_height in [8, 12, 16, 20]:
                for bar_width in [60, 80, 100, 120, 150]:
                    # Create horizontal bar template
                    bar_template = np.zeros((bar_height, bar_width), dtype=np.uint8)
                    border_thickness = max(1, bar_height // 4)

                    # Create border
                    cv2.rectangle(
                        bar_template,
                        (0, 0),
                        (bar_width - 1, bar_height - 1),
                        255,
                        border_thickness,
                    )

                    # Match template
                    result = cv2.matchTemplate(gray, bar_template, cv2.TM_CCOEFF_NORMED)
                    max_val = np.max(result)

                    if max_val > 0.4:
                        bar_confidences.append(max_val)

            # Also check for filled bars (common health/mana bars)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            horizontal_bars = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Look for horizontal bar-like shapes
                if 4 < aspect_ratio < 20 and 40 < w < 300 and 5 < h < 25:
                    horizontal_bars += 1

            if horizontal_bars > 1:
                bar_confidences.append(min(horizontal_bars / 5, 0.8))

            return max(bar_confidences) if bar_confidences else 0.0
        except Exception:
            return 0.0

    def _detect_minimap_regions(
        self, gray: np.ndarray, width: int, height: int
    ) -> float:
        """Detect minimap regions in corners with improved algorithm"""
        try:
            corner_size = min(200, width // 4, height // 4)
            confidence_scores = []

            # Check all four corners
            corners = [
                (gray[:corner_size, :corner_size], "top_left"),
                (gray[:corner_size, -corner_size:], "top_right"),
                (gray[-corner_size:, :corner_size], "bottom_left"),
                (gray[-corner_size:, -corner_size:], "bottom_right"),
            ]

            for corner_img, corner_name in corners:
                # Look for circular or square bounded regions
                edges = cv2.Canny(corner_img, 50, 150)
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if (
                        500 < area < corner_size * corner_size * 0.8
                    ):  # Reasonable minimap size
                        # Check if contour is roughly circular or square
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if 0.3 < circularity < 1.2:  # Reasonable shape
                                # Check for high detail density (maps have lots of detail)
                                mask = np.zeros(corner_img.shape, np.uint8)
                                cv2.drawContours(mask, [contour], -1, 255, -1)
                                detail_density = np.mean(cv2.bitwise_and(edges, mask))

                                if detail_density > 10:  # High detail suggests minimap
                                    confidence_scores.append(
                                        min(circularity * detail_density / 50, 1.0)
                                    )

            return max(confidence_scores) if confidence_scores else 0.0
        except Exception:
            return 0.0

    def _detect_hud_elements(self, gray: np.ndarray, color_image: np.ndarray) -> float:
        """Detect HUD elements like health counters, timers, scores"""
        try:
            confidence_factors = []

            # Detect number-like regions (scores, health values, timers)
            # Look for small rectangular regions with text
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            number_regions = 0
            small_ui_elements = 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h

                # Look for text-like rectangles
                if 0.2 < aspect_ratio < 5 and 100 < area < 2000:
                    # Check if it's likely text/numbers
                    roi = gray[y : y + h, x : x + w]
                    if np.std(roi) > 30:  # Has contrast (likely text)
                        number_regions += 1

                # Small UI elements (icons, buttons)
                elif 15 < w < 60 and 15 < h < 60 and area > 200:
                    small_ui_elements += 1

            if number_regions > 3:
                confidence_factors.append(min(number_regions / 10, 0.7))

            if small_ui_elements > 5:
                confidence_factors.append(min(small_ui_elements / 20, 0.5))

            # Check for semi-transparent overlays (common in games)
            overlay_confidence = self._detect_overlay_elements(color_image)
            if overlay_confidence > 0.3:
                confidence_factors.append(overlay_confidence * 0.6)

            return max(confidence_factors) if confidence_factors else 0.0
        except Exception:
            return 0.0

    def _detect_gaming_colors(self, hsv: np.ndarray) -> float:
        """Detect color patterns typical of gaming UIs"""
        try:
            # Gaming UIs often use specific color schemes
            confidence_factors = []

            # Check for vibrant colors common in games
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]

            # High saturation areas (vibrant UI elements)
            high_sat_ratio = np.sum(saturation > 180) / saturation.size
            if high_sat_ratio > 0.1:  # More than 10% vibrant colors
                confidence_factors.append(min(high_sat_ratio * 3, 0.6))

            # Check for specific gaming color ranges
            # Health bars (red/green), mana bars (blue), UI elements (gold/yellow)
            gaming_colors = [
                ([0, 100, 100], [10, 255, 255]),  # Red (health)
                ([170, 100, 100], [180, 255, 255]),  # Red (health, wrapped)
                ([40, 100, 100], [80, 255, 255]),  # Green (health)
                ([100, 100, 100], [130, 255, 255]),  # Blue (mana)
                ([15, 100, 100], [35, 255, 255]),  # Yellow/Gold (UI)
            ]

            gaming_color_pixels = 0
            for lower, upper in gaming_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                gaming_color_pixels += np.sum(mask > 0)

            gaming_color_ratio = gaming_color_pixels / (hsv.shape[0] * hsv.shape[1])
            if gaming_color_ratio > 0.05:
                confidence_factors.append(min(gaming_color_ratio * 10, 0.7))

            return max(confidence_factors) if confidence_factors else 0.0
        except Exception:
            return 0.0

    def _detect_crosshair(self, gray: np.ndarray, width: int, height: int) -> float:
        """Detect crosshair/reticle in center of screen"""
        try:
            # Check center region for crosshair patterns
            center_size = min(100, width // 10, height // 10)
            center_x, center_y = width // 2, height // 2

            center_region = gray[
                center_y - center_size : center_y + center_size,
                center_x - center_size : center_x + center_size,
            ]

            if center_region.size == 0:
                return 0.0

            # Create crosshair templates
            templates = []
            for size in [5, 7, 9, 11]:
                # Simple cross template
                template = np.zeros((size * 2 + 1, size * 2 + 1), dtype=np.uint8)
                cv2.line(
                    template, (0, size), (size * 2, size), 255, 1
                )  # Horizontal line
                cv2.line(template, (size, 0), (size, size * 2), 255, 1)  # Vertical line
                templates.append(template)

                # Dot with cross template
                template_dot = template.copy()
                cv2.circle(template_dot, (size, size), 2, 255, -1)
                templates.append(template_dot)

            max_confidence = 0.0
            for template in templates:
                if (
                    template.shape[0] <= center_region.shape[0]
                    and template.shape[1] <= center_region.shape[1]
                ):
                    result = cv2.matchTemplate(
                        center_region, template, cv2.TM_CCOEFF_NORMED
                    )
                    max_confidence = max(max_confidence, np.max(result))

            return max_confidence if max_confidence > 0.5 else 0.0
        except Exception:
            return 0.0

    def _detect_action_icons(self, image: np.ndarray) -> float:
        """Detect action bars with skill/ability icons"""
        try:
            # Convert to HSV for better icon detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Look for grid-like arrangements of icons
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find square/rectangular regions that could be icons
            icon_candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Icons are typically square or slightly rectangular
                if 0.7 < aspect_ratio < 1.4 and 20 < w < 80 and 20 < h < 80:
                    # Check if the region has good contrast (icon-like)
                    roi = gray[y : y + h, x : x + w]
                    if np.std(roi) > 25:  # Good contrast
                        icon_candidates.append((x, y, w, h))

            # Look for horizontal/vertical arrangements of icons
            if len(icon_candidates) < 3:
                return 0.0

            # Check for grid patterns
            horizontal_groups = []
            vertical_groups = []

            for i, (x1, y1, w1, h1) in enumerate(icon_candidates):
                h_group = [i]
                v_group = [i]

                for j, (x2, y2, w2, h2) in enumerate(icon_candidates[i + 1 :], i + 1):
                    # Check if horizontally aligned
                    if abs(y1 - y2) < 10 and abs(x1 - x2) < 100:
                        h_group.append(j)

                    # Check if vertically aligned
                    if abs(x1 - x2) < 10 and abs(y1 - y2) < 100:
                        v_group.append(j)

                if len(h_group) >= 3:
                    horizontal_groups.append(h_group)
                if len(v_group) >= 3:
                    vertical_groups.append(v_group)

            # Calculate confidence based on icon arrangements
            total_groups = len(horizontal_groups) + len(vertical_groups)
            if total_groups > 0:
                return min(total_groups * 0.3, 1.0)

            return 0.0
        except Exception:
            return 0.0

    def _detect_overlay_elements(self, image: np.ndarray) -> float:
        """Detect semi-transparent overlay elements common in games"""
        try:
            # Look for regions with consistent alpha-like blending effects
            # This is approximated by looking for regions with intermediate pixel values

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find regions with intermediate brightness (possible overlays)
            overlay_mask = (gray > 50) & (gray < 200)
            overlay_ratio = np.sum(overlay_mask) / overlay_mask.size

            # Check for rectangular overlay regions
            contours, _ = cv2.findContours(
                overlay_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            overlay_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Reasonable overlay size
                    # Check if roughly rectangular
                    x, y, w, h = cv2.boundingRect(contour)
                    extent = area / (w * h)
                    if extent > 0.7:  # Fairly rectangular
                        overlay_regions += 1

            if overlay_regions > 0 and overlay_ratio > 0.1:
                return min(overlay_regions * overlay_ratio * 5, 1.0)

            return 0.0
        except Exception:
            return 0.0

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
