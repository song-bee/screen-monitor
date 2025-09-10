#!/usr/bin/env python3
"""
Test script for vision detection functionality
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.core.capture.screen import ScreenCaptureManager
from asam.core.detection.analyzers.vision import VisionAnalyzer


async def test_vision_detection():
    """Test the vision detection system"""
    print("=== Testing Vision Detection System ===")

    # Initialize components
    config = {
        "capture_quality": 85,
        "exclude_menu_bar": True,
        "min_image_size": (100, 100),
        "enable_ad_detection": True,
        "enable_video_detection": True,
        "enable_game_detection": True,
        "video_motion_threshold": 0.3,
        "ad_template_threshold": 0.8,
        "game_ui_threshold": 0.6,
    }

    try:
        # Initialize screen capture
        capture_manager = ScreenCaptureManager(config)
        print("✓ Screen capture initialized")

        # Initialize vision analyzer
        vision_analyzer = VisionAnalyzer(config)
        print("✓ Vision analyzer initialized")

        # Get screen info
        screen_info = await capture_manager.get_screen_info()
        print(f"✓ Screen info: {screen_info}")

        print("\n=== Performing Detection Tests ===")

        # Capture multiple frames to test motion detection
        previous_results = []
        for i in range(5):
            print(f"\n--- Frame {i+1}/5 ---")

            # Capture screen
            start_time = time.time()
            screen_capture = await capture_manager.capture_screen()
            capture_time = time.time() - start_time

            print(
                f"Screen captured: {screen_capture.width}x{screen_capture.height} in {capture_time:.3f}s"
            )
            print(f"Active window: {screen_capture.active_window_title}")
            print(f"Active process: {screen_capture.active_process_name}")
            print(f"Capture source: {screen_capture.source}")

            # Analyze with vision detector
            analysis_start = time.time()
            result = await vision_analyzer.analyze(screen_capture)
            analysis_time = time.time() - analysis_start

            if result:
                print(f"✓ Detection result (analysis: {analysis_time:.3f}s):")
                print(f"  - Confidence: {result.confidence:.3f}")
                print(f"  - Category: {result.category.value}")
                print(f"  - Analyzer: {result.analyzer_type.value}")

                # Show evidence summary
                evidence_summary = {}
                for key, value in result.evidence.items():
                    if isinstance(value, (int, float)):
                        evidence_summary[key] = value
                    elif isinstance(value, list) and len(value) <= 3:
                        evidence_summary[key] = value
                    elif key in [
                        "detection_type",
                        "detection_methods",
                        "analysis_methods",
                    ]:
                        evidence_summary[key] = value

                if evidence_summary:
                    print(f"  - Key evidence: {json.dumps(evidence_summary, indent=4)}")

                previous_results.append(
                    {
                        "confidence": result.confidence,
                        "category": result.category.value,
                        "evidence_keys": list(result.evidence.keys()),
                    }
                )
            else:
                print("  - No significant detection")
                previous_results.append(None)

            # Wait between captures
            if i < 4:  # Don't wait after last capture
                print("  Waiting 2 seconds...")
                await asyncio.sleep(2)

        print("\n=== Detection Summary ===")

        # Analyze results
        valid_results = [r for r in previous_results if r is not None]
        if valid_results:
            avg_confidence = sum(r["confidence"] for r in valid_results) / len(
                valid_results
            )
            categories = [r["category"] for r in valid_results]
            print(f"Valid detections: {len(valid_results)}/5")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"Categories detected: {set(categories)}")

            # Check if motion detection is working
            motion_detections = sum(
                1
                for r in valid_results
                if "motion_ratio" in str(r.get("evidence_keys", []))
            )
            print(f"Motion detections: {motion_detections}/5")

            if motion_detections >= 2:
                print("✓ Motion detection appears to be working")
            else:
                print(
                    "⚠ Limited motion detection - this may be normal for static content"
                )
        else:
            print(
                "No detections found - this may be normal for non-entertainment content"
            )

        print("\n=== Test Complete ===")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_vision_detection())
    sys.exit(0 if success else 1)
