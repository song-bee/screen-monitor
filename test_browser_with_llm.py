#!/usr/bin/env python3
"""
ASAM Browser Integration with LLM Analysis Test

Tests browser extension with full LLM content analysis and verbose output.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.integrations.browser import BrowserExtensionManager
from asam.core.detection.analyzers.text import TextAnalyzer
from asam.core.detection.types import TextContent


def truncate_text(text, max_length=300):
    """Truncate text for display with proper formatting"""
    if not text:
        return "No content"

    text = text.strip()
    if len(text) <= max_length:
        return text

    return text[:max_length] + f"... ({len(text)} total chars)"


def format_llm_response(response_data):
    """Format LLM response for readable display"""
    if not response_data:
        return "No response"

    # Extract key information
    category = response_data.get('category', 'unknown')
    confidence = response_data.get('confidence', 0.0)
    reasoning = response_data.get('reasoning', 'No reasoning provided')
    keywords = response_data.get('keywords', [])
    subcategory = response_data.get('subcategory', 'unclear')

    formatted = f"""
    🎯 Category: {category.upper()}
    🔢 Confidence: {confidence:.2f}
    🏷️  Subcategory: {subcategory}
    🔑 Keywords: {', '.join(keywords) if keywords else 'None'}
    💭 Reasoning: {truncate_text(reasoning, 150)}"""

    return formatted


async def test_browser_with_llm_analysis():
    """Test browser integration with full LLM analysis"""

    print("🚀 Starting ASAM Browser + LLM Analysis Test...")
    print("=" * 80)

    try:
        # Initialize browser extension manager
        print("📱 Starting browser integration server...")
        browser_manager = BrowserExtensionManager()

        # Initialize text analyzer for LLM processing
        print("🧠 Initializing LLM text analyzer...")
        text_analyzer = TextAnalyzer()

        if not await text_analyzer.initialize():
            print("❌ Failed to initialize text analyzer (Ollama may not be running)")
            print("💡 Make sure Ollama is running with: brew services start ollama")
            return False

        if await browser_manager.start():
            print("✅ Browser integration server started!")

            # Display connection info
            server_info = browser_manager.get_server_info()
            print(f"🌐 Server: {server_info['endpoint']}")
            print(f"🔑 API Key: {browser_manager.server.api_key}")
            print(f"📊 Status: {server_info['status_endpoint']}")

            print("\n" + "=" * 80)
            print("🎯 VERBOSE BROWSER + LLM ANALYSIS READY")
            print("=" * 80)
            print("1. Install browser extension (see BROWSER_EXTENSION_SETUP.md)")
            print("2. Browse websites and watch detailed analysis below")
            print("3. Press Ctrl+C to stop")
            print("=" * 80)

            # Enhanced content callback with LLM analysis
            async def handle_browser_content_with_llm(browser_content):
                timestamp = datetime.now().strftime("%H:%M:%S")

                print(f"\n🔍 [{timestamp}] NEW CONTENT ANALYSIS")
                print("=" * 70)

                # Display basic info
                print(f"📄 Title: {browser_content.title}")
                print(f"🔗 URL: {browser_content.url}")
                print(f"🏷️  Tab ID: {browser_content.tab_id}")
                print(f"📊 Content Type: {browser_content.metadata.get('contentType', 'unknown')}")

                # Display content preview
                print(f"\n📖 WEBPAGE CONTENT ({len(browser_content.text_content)} chars):")
                print("-" * 50)
                content_preview = truncate_text(browser_content.text_content, 400)
                # Add indentation for better readability
                formatted_content = "\n".join([f"   {line}" for line in content_preview.split('\n')])
                print(formatted_content)
                print("-" * 50)

                # Display metadata
                if browser_content.metadata:
                    print(f"\n📊 METADATA:")
                    for key, value in browser_content.metadata.items():
                        if key not in ['contentType']:  # Already shown above
                            display_value = truncate_text(str(value), 80) if isinstance(value, str) else value
                            print(f"   {key}: {display_value}")

                # Convert to TextContent for LLM analysis
                print(f"\n🧠 LLM ANALYSIS:")
                print("-" * 30)

                try:
                    text_content = browser_manager.server.convert_to_text_content(browser_content)

                    # Perform LLM analysis
                    detection_result = await text_analyzer.analyze(text_content)

                    if detection_result:
                        print(f"✅ Analysis completed!")
                        print(f"🎯 Category: {detection_result.category.value}")
                        print(f"🔢 Confidence: {detection_result.confidence:.3f}")

                        # Display evidence details
                        if detection_result.evidence:
                            evidence = detection_result.evidence
                            print(f"\n📋 EVIDENCE:")

                            # LLM specific evidence
                            if 'reasoning' in evidence:
                                reasoning = truncate_text(evidence['reasoning'], 200)
                                print(f"   💭 Reasoning: {reasoning}")

                            if 'keywords' in evidence and evidence['keywords']:
                                keywords = evidence['keywords'][:10]  # Limit to first 10 keywords
                                print(f"   🔑 Keywords: {', '.join(keywords)}")

                            if 'subcategory' in evidence:
                                print(f"   🏷️  Subcategory: {evidence['subcategory']}")

                            if 'llm_category' in evidence:
                                print(f"   🤖 LLM Category: {evidence['llm_category']}")

                            # Technical details
                            print(f"   📏 Text Length: {evidence.get('text_length', 'unknown')}")
                            print(f"   📡 Source: {evidence.get('text_source', 'unknown')}")
                            print(f"   🧠 Model: {evidence.get('model_used', 'unknown')}")

                        # Determine action based on confidence
                        if detection_result.confidence >= 0.8:
                            action_color = "🔴"
                            action_text = "HIGH RISK - Would block/warn"
                        elif detection_result.confidence >= 0.6:
                            action_color = "🟡"
                            action_text = "MEDIUM RISK - Would monitor"
                        else:
                            action_color = "🟢"
                            action_text = "LOW RISK - Would allow"

                        print(f"\n{action_color} ACTION: {action_text}")

                    else:
                        print("❌ LLM analysis failed or returned no result")

                except Exception as e:
                    print(f"❌ Error during LLM analysis: {e}")
                    # Show traceback for debugging
                    import traceback
                    traceback.print_exc()

                print("\n" + "=" * 70)
                print(f"✅ Analysis complete for: {browser_content.title}")
                print("=" * 70)

            # Register the enhanced callback
            browser_manager.server.register_content_callback(handle_browser_content_with_llm)

            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Test interrupted by user")

        else:
            print("❌ Failed to start browser integration server")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if text_analyzer:
            await text_analyzer.cleanup()
        if browser_manager:
            await browser_manager.stop()
        print("✅ Verbose browser + LLM test completed")

    return True


async def test_single_content_analysis():
    """Test LLM analysis with sample content"""

    print("🧪 Testing LLM Analysis with Sample Content...")
    print("=" * 60)

    # Initialize text analyzer
    text_analyzer = TextAnalyzer()
    if not await text_analyzer.initialize():
        print("❌ Failed to initialize text analyzer")
        return

    # Sample content for testing
    test_contents = [
        {
            "title": "How to Build a React App",
            "url": "https://example.com/react-tutorial",
            "content": "Learn how to build modern React applications with hooks, state management, and component design patterns. This comprehensive tutorial covers everything from setup to deployment."
        },
        {
            "title": "Top 10 Funny Cat Videos",
            "url": "https://youtube.com/watch?v=cats",
            "content": "Check out these hilarious cat videos that will make you laugh! Funny cats playing, jumping, and being adorable. Don't miss these viral cat moments!"
        },
        {
            "title": "Stock Market Analysis Today",
            "url": "https://finance.example.com/analysis",
            "content": "Today's market analysis shows significant movement in tech stocks. Apple and Microsoft gained while Tesla declined. Economic indicators suggest continued volatility ahead."
        }
    ]

    for i, test_content in enumerate(test_contents, 1):
        print(f"\n📝 TEST CASE {i}:")
        print(f"Title: {test_content['title']}")
        print(f"URL: {test_content['url']}")
        print(f"Content: {truncate_text(test_content['content'], 100)}")

        # Create TextContent object
        text_content = TextContent(
            content=f"Page Title: {test_content['title']}\nURL: {test_content['url']}\nContent: {test_content['content']}",
            source="test_browser",
            timestamp=datetime.now(),
            metadata={"url": test_content['url'], "title": test_content['title']}
        )

        # Analyze with LLM
        try:
            result = await text_analyzer.analyze(text_content)
            if result:
                print(f"🎯 Result: {result.category.value} (confidence: {result.confidence:.3f})")
                if result.evidence and 'reasoning' in result.evidence:
                    print(f"💭 Reasoning: {truncate_text(result.evidence['reasoning'], 120)}")
            else:
                print("❌ No analysis result")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")

        print("-" * 40)

    await text_analyzer.cleanup()
    print("✅ Sample analysis test completed")


def main():
    """Main function"""
    print("ASAM Verbose Browser + LLM Analysis Test")
    print("Choose a test mode:")
    print("1. Live browser integration with verbose LLM analysis")
    print("2. Test LLM analysis with sample content")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(test_browser_with_llm_analysis())
    elif choice == "2":
        asyncio.run(test_single_content_analysis())
    else:
        print("Invalid choice. Exiting.")
        return False

    return True


if __name__ == "__main__":
    main()