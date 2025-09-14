#!/usr/bin/env python3
"""
Test script for focused motion detection
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


async def test_focused_motion_detection():
    """Test the focused motion detection system"""
    print("=== Testing Focused Motion Detection System ===")
    
    # Initialize components
    config = {
        "capture_quality": 85,
        "exclude_menu_bar": True,
        "min_image_size": (100, 100),
        "enable_ad_detection": True,
        "enable_video_detection": True,
        "enable_game_detection": True,
        "video_motion_threshold": 0.2,  # Lower threshold to catch more motion
        "ad_template_threshold": 0.8,
        "game_ui_threshold": 0.6,
    }
    
    try:
        # Initialize screen capture and vision analyzer
        capture_manager = ScreenCaptureManager(config)
        vision_analyzer = VisionAnalyzer(config)
        print("✓ Components initialized")
        
        # Get screen info
        screen_info = await capture_manager.get_screen_info()
        print(f"✓ Screen: {screen_info['width']}x{screen_info['height']}")
        
        print("\n=== Focused Motion Analysis ===")
        print("Instructions: Move your mouse or play a video to generate motion")
        print("The system will identify the most active areas and analyze them for video content\n")
        
        # Perform focused motion detection over multiple frames
        for frame_num in range(8):  # More frames for better temporal analysis
            print(f"--- Frame {frame_num + 1}/8 ---")
            
            # Capture screen
            start_time = time.time()
            screen_capture = await capture_manager.capture_screen()
            capture_time = time.time() - start_time
            
            print(f"Captured in {capture_time:.3f}s | Active: {screen_capture.active_process_name or 'Unknown'}")
            
            # Run vision analysis
            analysis_start = time.time()
            result = await vision_analyzer.analyze(screen_capture)
            analysis_time = time.time() - analysis_start
            
            if result:
                print(f"Analysis: {analysis_time:.3f}s | Confidence: {result.confidence:.3f} | Category: {result.category.value}")
                
                # Extract focused motion evidence
                evidence = result.evidence
                focused_evidence = {}
                
                # Show motion area analysis
                if 'motion_areas_count' in evidence:
                    focused_evidence['motion_areas_found'] = evidence['motion_areas_count']
                    
                if 'focused_motion_confidence' in evidence:
                    focused_evidence['focused_motion_confidence'] = evidence['focused_motion_confidence']
                    
                # Show other motion-related evidence
                motion_keys = ['motion_ratio', 'optical_flow_confidence', 'temporal_consistency', 
                              'region_motion_confidence', 'motion_pixels']
                for key in motion_keys:
                    if key in evidence:
                        focused_evidence[key] = evidence[key]
                
                if focused_evidence:
                    print(f"Motion Evidence: {json.dumps(focused_evidence, indent=2)}")
                    
                    # Highlight focused motion detection
                    if 'focused_motion_confidence' in focused_evidence:
                        fmc = focused_evidence['focused_motion_confidence']
                        if fmc > 0.6:
                            print(f"🎥 HIGH: Focused motion suggests strong video activity (confidence: {fmc:.3f})")
                        elif fmc > 0.4:
                            print(f"📺 MEDIUM: Focused motion suggests possible video content (confidence: {fmc:.3f})")
                        else:
                            print(f"👁️ LOW: Some focused motion detected (confidence: {fmc:.3f})")
                
                # Show other detection methods for comparison
                other_methods = []
                for method in ['ui_bars_confidence', 'hud_confidence', 'gaming_colors_confidence', 
                              'crosshair_confidence', 'action_icons_confidence']:
                    if method in evidence and evidence[method] > 0.3:
                        other_methods.append(f"{method}: {evidence[method]:.3f}")
                
                if other_methods:
                    print(f"Other detections: {', '.join(other_methods)}")
                    
            else:
                print("No detection result")
            
            # Wait between captures to allow for motion
            if frame_num < 7:
                print("Waiting 1.5s for motion...")
                await asyncio.sleep(1.5)
            
            print()
        
        print("=== Focused Motion Test Complete ===")
        print("\nTips for testing:")
        print("1. Play a video in a browser or media player")
        print("2. Move windows around on screen") 
        print("3. Scroll through content with motion")
        print("4. Try gaming content with UI animations")
        print("\nThe system should identify active rectangles and analyze them specifically for video characteristics.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_focused_motion_detection())
    sys.exit(0 if success else 1)