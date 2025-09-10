#!/usr/bin/env python3
"""
Test integration of vision detection with the main detection engine
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.config.manager import ConfigManager
from asam.core.capture.screen import ScreenCaptureManager
from asam.core.detection.engine import DetectionEngine
from asam.core.detection.types import AnalysisType


async def test_detection_engine_integration():
    """Test the detection engine with vision analyzer"""
    print("=== Testing Detection Engine Integration ===")

    try:
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print("✓ Configuration loaded")

        # Initialize detection engine
        detection_engine = DetectionEngine(config)

        # Initialize the detection engine
        if not await detection_engine.initialize():
            print("❌ Failed to initialize detection engine")
            return False
        print("✓ Detection engine initialized")

        # Initialize screen capture
        screen_config = {
            "capture_quality": 85,
            "exclude_menu_bar": True,
        }
        capture_manager = ScreenCaptureManager(screen_config)
        print("✓ Screen capture initialized")

        # Test detection cycle
        print("\n=== Running Detection Cycle ===")

        # Capture screen
        screen_capture = await capture_manager.capture_screen()
        print(f"Screen captured: {screen_capture.width}x{screen_capture.height}")
        print(f"Active window: {screen_capture.active_window_title}")

        # Run detection using the correct method
        aggregated_result = await detection_engine.analyze_screen_content(
            screen_capture=screen_capture
        )

        if aggregated_result:
            print("\n✓ Detection completed:")
            print(f"  - Overall confidence: {aggregated_result.overall_confidence:.3f}")
            print(f"  - Primary category: {aggregated_result.primary_category.value}")
            print(
                f"  - Recommended action: {aggregated_result.recommended_action.value}"
            )
            print(f"  - Analysis duration: {aggregated_result.analysis_duration_ms}ms")
            print(
                f"  - Individual results: {len(aggregated_result.individual_results)}"
            )

            # Show individual analyzer results
            for result in aggregated_result.individual_results:
                print(
                    f"    - {result.analyzer_type.value}: {result.confidence:.3f} ({result.category.value})"
                )

            # Check if vision analyzer was included
            vision_results = [
                r
                for r in aggregated_result.individual_results
                if r.analyzer_type == AnalysisType.VISION
            ]
            if vision_results:
                print(
                    f"\n✓ Vision analyzer contributed with confidence: {vision_results[0].confidence:.3f}"
                )

                # Show some vision evidence if available
                vision_evidence = vision_results[0].evidence
                key_evidence = {
                    k: v
                    for k, v in vision_evidence.items()
                    if k
                    in [
                        "motion_ratio",
                        "detection_methods",
                        "analysis_methods",
                        "confidence_factors",
                    ]
                }
                if key_evidence:
                    print(f"    Vision evidence: {key_evidence}")
            else:
                print("⚠ No vision analysis results found")

            print(
                f"\n{'🔴' if aggregated_result.is_entertainment else '🟢'} Content assessment: {'Entertainment detected' if aggregated_result.is_entertainment else 'Productive/safe content'}"
            )

        else:
            print("❌ No detection results returned")
            return False

        print("\n=== Integration Test Complete ===")

        # Cleanup
        await detection_engine.cleanup()
        print("✓ Detection engine cleanup complete")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_detection_engine_integration())
    sys.exit(0 if success else 1)
