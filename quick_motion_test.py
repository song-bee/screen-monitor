#!/usr/bin/env python3
"""
Quick test for motion detection performance
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.core.capture.screen import ScreenCaptureManager
from asam.core.detection.analyzers.vision import VisionAnalyzer


async def quick_test():
    """Quick performance test"""
    print("=== Quick Motion Test ===")
    
    config = {
        "capture_quality": 85,
        "exclude_menu_bar": True,
        "video_motion_threshold": 0.2,
    }
    
    try:
        capture_manager = ScreenCaptureManager(config)
        vision_analyzer = VisionAnalyzer(config)
        print("✓ Initialized")
        
        # Single frame test
        print("Capturing screen...")
        start_time = time.time()
        screen_capture = await capture_manager.capture_screen()
        capture_time = time.time() - start_time
        print(f"✓ Capture: {capture_time:.3f}s")
        
        print("Running vision analysis...")
        analysis_start = time.time()
        result = await vision_analyzer.analyze(screen_capture)
        analysis_time = time.time() - analysis_start
        print(f"✓ Analysis: {analysis_time:.3f}s")
        
        if result:
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Category: {result.category.value}")
            
            # Show key evidence
            evidence = result.evidence
            key_evidence = {}
            for key in ['focused_motion_confidence', 'motion_areas_count', 'motion_ratio']:
                if key in evidence:
                    key_evidence[key] = evidence[key]
            
            if key_evidence:
                print(f"Evidence: {key_evidence}")
        else:
            print("No result")
            
        print("✓ Test complete")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)