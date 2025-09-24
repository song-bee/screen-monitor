#!/usr/bin/env python3
"""
Ultra-Verbose ASAM Browser + LLM Analysis Test

Shows every detail of the analysis process including:
- Raw content extraction
- Complete LLM prompts
- Full LLM responses
- Step-by-step processing
- Performance timing
- Internal data structures
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.core.detection.analyzers.text import TextAnalyzer
from asam.core.detection.types import TextContent
from asam.integrations.browser import BrowserExtensionManager


def setup_ultra_verbose_logging():
    """Configure logging to show ALL details"""

    # Create custom formatter for ultra-verbose output
    class UltraVerboseFormatter(logging.Formatter):
        def format(self, record):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[
                :-3
            ]  # Include milliseconds

            # Color coding for different log levels
            colors = {
                "DEBUG": "\033[36m",  # Cyan
                "INFO": "\033[32m",  # Green
                "WARNING": "\033[33m",  # Yellow
                "ERROR": "\033[31m",  # Red
                "CRITICAL": "\033[35m",  # Magenta
            }
            reset_color = "\033[0m"

            color = colors.get(record.levelname, "")
            level = f"{color}[{record.levelname:<8}]{reset_color}"

            # Format module name for readability
            module_name = (
                record.name.split(".")[-1] if "." in record.name else record.name
            )
            module_display = f"{module_name:<15}"

            return f"{timestamp} | {level} | {module_display} | {record.getMessage()}"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with ultra-verbose formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(UltraVerboseFormatter())
    root_logger.addHandler(console_handler)

    # Set specific loggers to DEBUG
    logging.getLogger("asam.core.detection.analyzers.text").setLevel(logging.DEBUG)
    logging.getLogger("asam.integrations.browser").setLevel(logging.DEBUG)


def print_section_header(title, char="=", width=100):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_subsection(title, char="-", width=80):
    """Print a formatted subsection header"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def truncate_with_info(text, max_length=500):
    """Truncate text but show info about truncation"""
    if not text:
        return "No content", 0

    original_length = len(text)
    if original_length <= max_length:
        return text, original_length

    truncated = (
        text[:max_length]
        + f"\n... [TRUNCATED: {original_length - max_length} more characters]"
    )
    return truncated, original_length


async def test_ultra_verbose_analysis():
    """Ultra-verbose test with complete information display"""

    print_section_header("🚀 ASAM ULTRA-VERBOSE ANALYSIS TEST 🚀")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup ultra-verbose logging
    print_subsection("Setting up ultra-verbose logging system")
    setup_ultra_verbose_logging()
    print("✅ Ultra-verbose logging configured")

    # Test content samples
    test_samples = [
        {
            "name": "Gaming Content",
            "title": "EPIC Fortnite Battle Royale Victory Guide - How to Get Your First Win!",
            "url": "https://gaming.youtube.com/watch?v=fortnite_guide",
            "content": """🔥 ULTIMATE FORTNITE GUIDE! 🔥

Hey gamers! Welcome back to my channel! Today we're going to master Fortnite Battle Royale and get you that Victory Royale!

🎮 WHAT WE'LL COVER:
- Best landing spots for beginners and pros
- Building techniques that will blow your mind
- Weapon tier list - which guns to pick up
- End game strategies for clutch wins
- Secret tricks the pros don't want you to know!

💥 EPIC MOMENTS FROM LAST STREAM:
Yesterday's stream was INSANE! We got 5 Victory Royales in a row! The chat was going crazy with "POG" and "LET'S GOOO!"

Don't forget to SMASH that like button, subscribe for more gaming content, and hit the bell for notifications! Also check out my Twitch for live streams every night!

Drop your epic win stories in the comments below! 🚀

#Fortnite #Gaming #BattleRoyale #EpicGames #VictoryRoyale""",
            "expected_category": "entertainment",
            "content_type": "gaming",
        },
        {
            "name": "Educational Content",
            "title": "Advanced Python Programming: Async/Await and Concurrency Patterns",
            "url": "https://education.example.com/python-async-tutorial",
            "content": """Advanced Python Programming: Mastering Asynchronous Programming

In this comprehensive tutorial, we'll explore advanced concepts in Python's asynchronous programming paradigm.

Learning Objectives:
• Understand the asyncio event loop architecture
• Master async/await syntax and patterns
• Implement concurrent programming solutions
• Handle exceptions in asynchronous code
• Optimize performance with proper concurrency

Prerequisites:
- Solid understanding of Python fundamentals
- Familiarity with functions and decorators
- Basic knowledge of threading concepts

Course Structure:
1. Introduction to Asynchronous Programming
2. The asyncio Module Deep Dive
3. Implementing Async Patterns
4. Real-World Applications and Best Practices
5. Performance Analysis and Optimization

This tutorial is designed for intermediate to advanced Python developers looking to enhance their skills in concurrent programming. By the end, you'll be able to build scalable, high-performance applications using Python's async capabilities.

Code examples and exercises are provided throughout to reinforce learning concepts.""",
            "expected_category": "productive",
            "content_type": "education",
        },
        {
            "name": "Social Media Content",
            "title": "OMG! Celebrity Drama Explodes on Twitter! 🔥",
            "url": "https://twitter.com/gossip_central/status/123456789",
            "content": """OMG YOU GUYS!!! 😱😱😱

The drama is UNREAL right now! Did you see what just happened on Twitter??

✨ BREAKING: Famous actor just posted the most SAVAGE response to the haters! The quote tweets are going VIRAL!

💀 The replies are BRUTAL:
"Not you coming for her like that 💅"
"THE SHADE IS IMMACULATE"
"Main character energy ✨"
"This is why I stan 😍"

🔥 It's trending #1 worldwide with 2M tweets in the last hour! Even brands are getting in on the drama!

The memes are already LEGENDARY! My timeline is chaos and I'm LIVING for it! 📱✨

WHO'S SIDE ARE YOU ON?? Drop your hot takes in the replies!

This is going to be ALL OVER TikTok tomorrow! 📺🍿

#Drama #Twitter #Viral #Celebrity #Trending #SocialMedia""",
            "expected_category": "entertainment",
            "content_type": "social_media",
        },
    ]

    # Initialize components
    print_subsection("Initializing ASAM components")
    browser_manager = BrowserExtensionManager()
    text_analyzer = TextAnalyzer()

    try:
        # Component initialization with timing
        init_start = datetime.now()

        print("🔧 Initializing TextAnalyzer...")
        if not await text_analyzer.initialize():
            print("❌ CRITICAL ERROR: Text analyzer initialization failed!")
            print("💡 Ensure Ollama is running: brew services start ollama")
            return

        print("🔧 Initializing BrowserExtensionManager...")
        if not await browser_manager.start():
            print("❌ CRITICAL ERROR: Browser manager initialization failed!")
            return

        init_time = (datetime.now() - init_start).total_seconds()
        print(f"✅ All components initialized in {init_time:.2f}s")

        # Process each test sample
        for i, sample in enumerate(test_samples, 1):
            print_section_header(f"TEST CASE {i}: {sample['name'].upper()}", "=", 120)

            print("📊 Test Sample Information:")
            print(f"   • Name: {sample['name']}")
            print(f"   • Expected Category: {sample['expected_category']}")
            print(f"   • Content Type: {sample['content_type']}")
            print("   • Expected vs Actual: Will compare results")

            print_subsection("📄 ORIGINAL CONTENT ANALYSIS")

            print(f"🏷️  Title: {sample['title']}")
            print(f"🔗 URL: {sample['url']}")

            content_display, original_length = truncate_with_info(
                sample["content"], 800
            )
            print(f"📝 Content Length: {original_length} characters")
            print("📖 Full Content:")
            print("─" * 80)
            # Add line numbers and indentation for better readability
            lines = content_display.split("\n")
            for line_num, line in enumerate(lines, 1):
                print(f"{line_num:3d} │ {line}")
            print("─" * 80)

            print_subsection("🧠 LLM ANALYSIS PIPELINE")

            # Create TextContent object
            combined_text = f"Page Title: {sample['title']}\nURL: {sample['url']}\nContent: {sample['content']}"

            text_content = TextContent(
                content=combined_text,
                source="ultra_verbose_test",
                timestamp=datetime.now(),
                metadata={
                    "url": sample["url"],
                    "title": sample["title"],
                    "content_type": sample["content_type"],
                    "test_case": i,
                    "expected_category": sample["expected_category"],
                },
            )

            print("📦 TextContent Object Created:")
            print(f"   • Total content length: {len(combined_text)} chars")
            print(f"   • Source: {text_content.source}")
            print(f"   • Timestamp: {text_content.timestamp}")
            print(f"   • Metadata keys: {list(text_content.metadata.keys())}")

            # Perform analysis with timing
            analysis_start = datetime.now()
            print(
                f"\n⏱️  Starting LLM analysis at {analysis_start.strftime('%H:%M:%S.%f')[:-3]}"
            )

            try:
                print("\n🔄 Beginning detailed analysis pipeline...")
                detection_result = await text_analyzer.analyze(text_content)

                analysis_end = datetime.now()
                total_analysis_time = (
                    analysis_end - analysis_start
                ).total_seconds() * 1000

                print_subsection("📋 ANALYSIS RESULTS")

                if detection_result:
                    print("✅ Analysis completed successfully!")
                    print(f"⏱️  Total analysis time: {total_analysis_time:.1f}ms")

                    # Main results
                    print("\n🎯 CLASSIFICATION RESULTS:")
                    print(f"   • Category: {detection_result.category.value.upper()}")
                    print(
                        f"   • Confidence: {detection_result.confidence:.4f} ({detection_result.confidence*100:.1f}%)"
                    )
                    print(f"   • Analyzer Type: {detection_result.analyzer_type.value}")
                    print(f"   • Timestamp: {detection_result.timestamp}")

                    # Compare with expected
                    expected_match = (
                        detection_result.category.value == sample["expected_category"]
                    )
                    match_indicator = "✅ CORRECT" if expected_match else "❌ INCORRECT"
                    print(
                        f"   • Expected vs Actual: {sample['expected_category']} vs {detection_result.category.value} ({match_indicator})"
                    )

                    # Detailed evidence analysis
                    if detection_result.evidence:
                        evidence = detection_result.evidence
                        print("\n🔍 DETAILED EVIDENCE ANALYSIS:")

                        # LLM-specific evidence
                        llm_category = evidence.get("llm_category", "unknown")
                        subcategory = evidence.get("subcategory", "unclear")
                        print(f"   • LLM Raw Category: '{llm_category}'")
                        print(f"   • Subcategory: '{subcategory}'")
                        print(
                            f"   • Model Used: {evidence.get('model_used', 'unknown')}"
                        )

                        # Keywords analysis
                        keywords = evidence.get("keywords", [])
                        print(f"   • Keywords Found: {len(keywords)} total")
                        if keywords:
                            print(f"     Keywords: {', '.join(keywords)}")
                        else:
                            print("     Keywords: None identified")

                        # Reasoning analysis
                        reasoning = evidence.get("reasoning", "No reasoning provided")
                        reasoning_display, reasoning_length = truncate_with_info(
                            reasoning, 400
                        )
                        print(f"   • Reasoning Length: {reasoning_length} characters")
                        print("   • LLM Reasoning:")
                        reasoning_lines = reasoning_display.split("\n")
                        for line in reasoning_lines:
                            print(f"     │ {line}")

                        # Technical details
                        print("\n🔧 TECHNICAL DETAILS:")
                        print(
                            f"   • Original text length: {evidence.get('text_length', 'unknown')}"
                        )
                        print(
                            f"   • Text source: {evidence.get('text_source', 'unknown')}"
                        )

                        # Print all evidence keys for debugging
                        print(f"   • All evidence keys: {list(evidence.keys())}")

                    # Metadata analysis
                    if detection_result.metadata:
                        print("\n📊 RESULT METADATA:")
                        for key, value in detection_result.metadata.items():
                            print(f"   • {key}: {value}")

                    # Decision logic analysis
                    print("\n⚖️  DECISION ANALYSIS:")
                    confidence_level = (
                        "HIGH"
                        if detection_result.confidence >= 0.8
                        else "MEDIUM" if detection_result.confidence >= 0.6 else "LOW"
                    )
                    print(f"   • Confidence Level: {confidence_level}")

                    if detection_result.category.value == "entertainment":
                        if detection_result.confidence >= 0.8:
                            action = "🔴 BLOCK - High confidence entertainment content"
                        elif detection_result.confidence >= 0.6:
                            action = "🟡 WARN - Medium confidence entertainment content"
                        else:
                            action = "🟢 MONITOR - Low confidence, continue watching"
                    elif detection_result.category.value == "productive":
                        action = "✅ ALLOW - Productive content detected"
                    else:
                        action = "🤔 UNKNOWN - Unclear content, apply default policy"

                    print(f"   • Recommended Action: {action}")

                else:
                    analysis_time = (
                        datetime.now() - analysis_start
                    ).total_seconds() * 1000
                    print(f"❌ Analysis FAILED after {analysis_time:.1f}ms")
                    print("   • No detection result returned")
                    print("   • Check logs above for error details")

            except Exception as e:
                analysis_time = (datetime.now() - analysis_start).total_seconds() * 1000
                print(f"❌ Analysis CRASHED after {analysis_time:.1f}ms")
                print(f"   • Error: {str(e)}")
                print(f"   • Error Type: {type(e).__name__}")

                # Full traceback for debugging
                import traceback

                print("\n🐛 FULL ERROR TRACEBACK:")
                print("─" * 80)
                traceback.print_exc()
                print("─" * 80)

            # Separator between test cases
            if i < len(test_samples):
                print("\n⏳ Waiting 2 seconds before next test case...")
                await asyncio.sleep(2)

        print_section_header("🎉 ALL ULTRA-VERBOSE TESTS COMPLETED 🎉", "=", 120)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print("\n💥 CRITICAL ERROR in test execution:")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        import traceback

        print("Full traceback:")
        traceback.print_exc()

    finally:
        print_subsection("🧹 Cleanup Process")
        cleanup_start = datetime.now()

        print("🔄 Cleaning up TextAnalyzer...")
        await text_analyzer.cleanup()

        print("🔄 Cleaning up BrowserExtensionManager...")
        await browser_manager.stop()

        cleanup_time = (datetime.now() - cleanup_start).total_seconds()
        print(f"✅ Cleanup completed in {cleanup_time:.2f}s")

        print_section_header("🏁 ULTRA-VERBOSE TEST SESSION COMPLETE 🏁")


def main():
    """Run the ultra-verbose test"""
    try:
        asyncio.run(test_ultra_verbose_analysis())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
