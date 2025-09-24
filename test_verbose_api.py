#!/usr/bin/env python3
"""
Test verbose API interaction with detailed content and LLM response display
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.core.detection.analyzers.text import TextAnalyzer
from asam.core.detection.types import TextContent
from asam.integrations.browser import BrowserExtensionManager


async def test_verbose_api_with_llm():
    """Test API with verbose content display and LLM analysis"""

    print("🚀 ASAM Verbose API + LLM Analysis Test")
    print("=" * 80)

    # Sample test contents with different types
    test_contents = [
        {
            "title": "Gaming Tutorial - How to Master Fortnite",
            "url": "https://gaming.example.com/fortnite-guide",
            "content": """Ultimate Fortnite Battle Royale guide! Learn the best strategies to win games, master building techniques, find the best weapons, and dominate your opponents. Tips for beginners and advanced players. Epic Victory Royale awaits!

            Key strategies:
            - Land in popular spots for action
            - Master the building mechanics
            - Learn weapon tier lists
            - Practice your aim in creative mode
            - Watch top streamers for pro tips

            Don't forget to subscribe for more gaming content and hit the notification bell!""",
            "contentType": "gaming",
        },
        {
            "title": "Python Web Development Tutorial",
            "url": "https://learn.example.com/python-tutorial",
            "content": """Learn Python web development with Django and Flask. This comprehensive tutorial covers:

            - Setting up your development environment
            - Creating your first web application
            - Database integration with SQLAlchemy
            - User authentication and authorization
            - Deployment strategies and best practices
            - Testing methodologies

            By the end of this course, you'll be able to build production-ready web applications using Python frameworks.""",
            "contentType": "education",
        },
        {
            "title": "Breaking: Celebrity Drama Unfolds",
            "url": "https://entertainment.example.com/celebrity-news",
            "content": """EXCLUSIVE: Hollywood drama as famous actor spotted with mystery person! Social media erupts with speculation and fan theories.

            The photos show them together at a luxury restaurant, sparking dating rumors. Fans are going crazy on Twitter and Instagram with hashtags trending worldwide.

            This comes just weeks after the controversial breakup that shocked the entertainment world. Stay tuned for more juicy details and exclusive photos!""",
            "contentType": "entertainment",
        },
    ]

    # Start browser integration server
    browser_manager = BrowserExtensionManager()
    text_analyzer = TextAnalyzer()

    try:
        # Initialize components
        print("🔧 Initializing components...")
        if not await text_analyzer.initialize():
            print("❌ Failed to initialize text analyzer (check Ollama)")
            return

        if not await browser_manager.start():
            print("❌ Failed to start browser server")
            return

        print("✅ All components initialized")
        print("🌐 Server running at: http://localhost:8888")

        # Test each content sample
        for i, test_content in enumerate(test_contents, 1):
            print(f"\n{'='*80}")
            print(f"🧪 TEST CASE {i}: {test_content['contentType'].upper()} CONTENT")
            print(f"{'='*80}")

            # Display original content
            print("📄 ORIGINAL CONTENT:")
            print(f"   Title: {test_content['title']}")
            print(f"   URL: {test_content['url']}")
            print(f"   Type: {test_content['contentType']}")
            print(f"   Content Length: {len(test_content['content'])} characters")

            print("\n📝 FULL CONTENT:")
            print("-" * 50)
            # Show content with proper indentation
            content_lines = test_content["content"].strip().split("\n")
            for line in content_lines:
                print(f"   {line}")
            print("-" * 50)

            # Send via API
            print("\n📡 SENDING TO ASAM API...")
            api_payload = {
                "url": test_content["url"],
                "title": test_content["title"],
                "content": test_content["content"],
                "tabId": f"test_{i}",
                "browserType": "test",
                "metadata": {
                    "contentType": test_content["contentType"],
                    "source": "api_test",
                    "timestamp": datetime.now().isoformat(),
                },
            }

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        "http://localhost:8888/api/content",
                        json=api_payload,
                        headers={"X-API-Key": "asam-browser-integration"},
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"   ✅ API Response: {result['status']}")
                            print(f"   📅 Timestamp: {result['timestamp']}")
                        else:
                            print(f"   ❌ API Error: {response.status}")
                            continue
                except Exception as e:
                    print(f"   ❌ API Request failed: {e}")
                    continue

            # Manual LLM Analysis for comparison
            print("\n🧠 LLM ANALYSIS:")
            print("-" * 40)

            try:
                # Create TextContent object
                combined_text = f"Page Title: {test_content['title']}\\nURL: {test_content['url']}\\nContent: {test_content['content']}"

                text_content = TextContent(
                    content=combined_text,
                    source="api_test",
                    timestamp=datetime.now(),
                    metadata={
                        "url": test_content["url"],
                        "title": test_content["title"],
                        "contentType": test_content["contentType"],
                    },
                )

                # Perform analysis
                detection_result = await text_analyzer.analyze(text_content)

                if detection_result:
                    print(f"   🎯 CATEGORY: {detection_result.category.value.upper()}")
                    print(f"   🔢 CONFIDENCE: {detection_result.confidence:.3f}")

                    if detection_result.evidence:
                        evidence = detection_result.evidence

                        # Show LLM reasoning
                        if "reasoning" in evidence:
                            reasoning = evidence["reasoning"]
                            print("\\n   💭 LLM REASONING:")
                            # Format reasoning with proper indentation
                            reasoning_lines = reasoning.split(". ")
                            for line in reasoning_lines:
                                if line.strip():
                                    print(f"      • {line.strip()}")

                        # Show keywords
                        if "keywords" in evidence and evidence["keywords"]:
                            keywords = evidence["keywords"]
                            print(
                                f"\\n   🔑 KEYWORDS IDENTIFIED: {', '.join(keywords)}"
                            )

                        # Show subcategory
                        if "subcategory" in evidence:
                            print(f"   🏷️  SUBCATEGORY: {evidence['subcategory']}")

                        # Show LLM technical details
                        print("\\n   🤖 TECHNICAL DETAILS:")
                        print(f"      Model: {evidence.get('model_used', 'unknown')}")
                        print(
                            f"      Input length: {evidence.get('text_length', 'unknown')} chars"
                        )
                        print(f"      Source: {evidence.get('text_source', 'unknown')}")

                    # Action recommendation
                    if detection_result.category.value == "entertainment":
                        if detection_result.confidence >= 0.8:
                            action = "🔴 HIGH RISK - WOULD BLOCK"
                        elif detection_result.confidence >= 0.6:
                            action = "🟡 MEDIUM RISK - WOULD WARN"
                        else:
                            action = "🟢 LOW RISK - WOULD MONITOR"
                    else:
                        action = "✅ PRODUCTIVE CONTENT - WOULD ALLOW"

                    print(f"\\n   ⚡ RECOMMENDED ACTION: {action}")

                else:
                    print("   ❌ No analysis result returned")

            except Exception as e:
                print(f"   ❌ LLM Analysis failed: {e}")
                import traceback

                traceback.print_exc()

            print(f"\\n✅ Test case {i} completed")
            print(f"{'='*80}")

            # Wait between tests
            if i < len(test_contents):
                print("⏳ Waiting 2 seconds before next test...")
                await asyncio.sleep(2)

        print("\\n🎉 ALL VERBOSE TESTS COMPLETED!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\\n🧹 Cleaning up...")
        await text_analyzer.cleanup()
        await browser_manager.stop()
        print("✅ Verbose API test completed")


if __name__ == "__main__":
    asyncio.run(test_verbose_api_with_llm())
