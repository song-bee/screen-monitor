#!/usr/bin/env python3
"""
Test focused motion area detection with fewer frames
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.core.capture.screen import ScreenCaptureManager
from asam.core.detection.analyzers.vision import VisionAnalyzer


async def test_motion_areas():
    """Test motion area detection with focused analysis"""
    print("=== Testing Motion Area Detection ===")
    
    config = {
        "capture_quality": 85,
        "exclude_menu_bar": True,
        "video_motion_threshold": 0.2,
    }
    
    try:
        capture_manager = ScreenCaptureManager(config)
        vision_analyzer = VisionAnalyzer(config)
        print("✓ Components initialized")
        
        print("\n=== Motion Area Analysis (3 frames) ===")
        print("Move windows or play videos to see focused motion detection...\n")
        
        for i in range(3):
            print(f"--- Frame {i+1}/3 ---")
            
            # Capture and analyze
            start_time = time.time()
            screen_capture = await capture_manager.capture_screen()
            capture_time = time.time() - start_time
            
            analysis_start = time.time()
            result = await vision_analyzer.analyze(screen_capture)
            analysis_time = time.time() - analysis_start
            
            total_time = capture_time + analysis_time
            print(f"Times: capture={capture_time:.2f}s, analysis={analysis_time:.2f}s, total={total_time:.2f}s")
            
            if result:
                conf = result.confidence
                cat = result.category.value
                print(f"Result: {conf:.3f} confidence, category={cat}")
                
                # Extract motion-specific evidence
                evidence = result.evidence
                motion_evidence = []
                
                if 'motion_areas_count' in evidence:
                    count = evidence['motion_areas_count']
                    motion_evidence.append(f"motion_areas={count}")
                    
                if 'focused_motion_confidence' in evidence:
                    fmc = evidence['focused_motion_confidence']
                    motion_evidence.append(f"focused_motion={fmc:.3f}")
                    
                if 'motion_ratio' in evidence:
                    mr = evidence['motion_ratio']
                    motion_evidence.append(f"motion_ratio={mr:.3f}")
                    
                if 'optical_flow_confidence' in evidence:
                    ofc = evidence['optical_flow_confidence']
                    motion_evidence.append(f"optical_flow={ofc:.3f}")
                
                if motion_evidence:
                    print(f"Motion: {', '.join(motion_evidence)}")
                
                # Highlight focused motion detection results
                if 'focused_motion_confidence' in evidence:
                    fmc = evidence['focused_motion_confidence']
                    if fmc > 0.5:
                        print("🎥 STRONG focused motion detected - likely video content!")
                    elif fmc > 0.3:
                        print("📺 MODERATE focused motion detected")
                    else:
                        print("👁️ WEAK focused motion")
                else:
                    print("No focused motion areas found")
                    
            else:
                print("No detection result")
            
            if i < 2:  # Don't wait after last frame
                await asyncio.sleep(2)
            print()
        
        print("=== Motion Area Test Complete ===")
        print("\nThe focused motion detection system:")
        print("1. Finds rectangular areas with the most motion")
        print("2. Analyzes those areas specifically for video characteristics")
        print("3. Uses optical flow, color dynamics, and temporal consistency")
        print("4. Provides higher accuracy for video detection in active regions")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_motion_areas())
    sys.exit(0 if success else 1)