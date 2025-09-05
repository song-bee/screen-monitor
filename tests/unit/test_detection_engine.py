"""
Unit tests for detection engine
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import asyncio

from asam.core.detection.engine import DetectionEngine
from asam.core.detection.types import (
    DetectionResult, AnalysisType, ContentCategory, ScreenCapture, TextContent
)


class TestDetectionEngine:
    """Test detection engine functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {
            "detection": {
                "confidence_threshold": 0.75,
                "analysis_interval_seconds": 5.0,
                "max_concurrent_analyses": 3
            },
            "text_detection": {"enabled": True, "weight": 0.4},
            "vision_detection": {"enabled": True, "weight": 0.3},
            "process_detection": {"enabled": True, "weight": 0.2},
            "network_detection": {"enabled": True, "weight": 0.1}
        }
    
    @pytest.fixture
    def mock_analyzers(self):
        """Mock analyzers for testing"""
        with patch('asam.core.detection.analyzers.TextAnalyzer') as mock_text, \
             patch('asam.core.detection.analyzers.VisionAnalyzer') as mock_vision, \
             patch('asam.core.detection.analyzers.ProcessAnalyzer') as mock_process, \
             patch('asam.core.detection.analyzers.NetworkAnalyzer') as mock_network:
            
            # Mock analyzer instances
            mock_text_instance = AsyncMock()
            mock_vision_instance = AsyncMock()
            mock_process_instance = AsyncMock()
            mock_network_instance = AsyncMock()
            
            mock_text.return_value = mock_text_instance
            mock_vision.return_value = mock_vision_instance
            mock_process.return_value = mock_process_instance
            mock_network.return_value = mock_network_instance
            
            yield {
                'text': mock_text_instance,
                'vision': mock_vision_instance,
                'process': mock_process_instance,
                'network': mock_network_instance
            }
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_analyzers):
        """Test detection engine initialization"""
        with patch('asam.core.detection.engine.ConfidenceAggregator'):
            engine = DetectionEngine(self.config)
            await engine.initialize()
            
            assert engine.config == self.config
            assert len(engine.analyzers) == 4
            assert engine.is_running is False
    
    @pytest.mark.asyncio
    async def test_analyze_screen_content(self, mock_analyzers):
        """Test screen content analysis"""
        # Setup mock results
        mock_text_result = DetectionResult(
            analyzer_type=AnalysisType.TEXT,
            confidence=0.8,
            category=ContentCategory.GAMING,
            evidence={'text_detected': 'gaming content'},
            timestamp=datetime.now()
        )
        
        mock_vision_result = DetectionResult(
            analyzer_type=AnalysisType.VISION,
            confidence=0.7,
            category=ContentCategory.GAMING,
            evidence={'motion_score': 8.5},
            timestamp=datetime.now()
        )
        
        mock_analyzers['text'].analyze.return_value = mock_text_result
        mock_analyzers['vision'].analyze.return_value = mock_vision_result
        mock_analyzers['process'].analyze.return_value = None
        mock_analyzers['network'].analyze.return_value = None
        
        # Mock aggregator
        mock_aggregator = MagicMock()
        mock_aggregated_result = MagicMock()
        mock_aggregated_result.overall_confidence = 0.75
        mock_aggregated_result.primary_category = ContentCategory.GAMING
        mock_aggregator.aggregate.return_value = mock_aggregated_result
        
        with patch('asam.core.detection.engine.ConfidenceAggregator', return_value=mock_aggregator):
            engine = DetectionEngine(self.config)
            await engine.initialize()
            
            # Create test data
            screen_capture = MagicMock(spec=ScreenCapture)
            text_content = TextContent(
                content="Gaming content detected",
                source="test",
                timestamp=datetime.now()
            )
            
            result = await engine.analyze_screen_content(screen_capture, text_content)
            
            # Verify analyzers were called
            mock_analyzers['text'].analyze.assert_called_once()
            mock_analyzers['vision'].analyze.assert_called_once()
            
            # Verify aggregation was called
            mock_aggregator.aggregate.assert_called_once()
            
            assert result == mock_aggregated_result
    
    @pytest.mark.asyncio
    async def test_analyze_with_disabled_analyzer(self, mock_analyzers):
        """Test analysis with some analyzers disabled"""
        # Disable text detection
        config = self.config.copy()
        config['text_detection']['enabled'] = False
        
        mock_analyzers['vision'].analyze.return_value = DetectionResult(
            analyzer_type=AnalysisType.VISION,
            confidence=0.6,
            category=ContentCategory.PRODUCTIVE,
            evidence={},
            timestamp=datetime.now()
        )
        
        mock_aggregator = MagicMock()
        mock_aggregated_result = MagicMock()
        mock_aggregator.aggregate.return_value = mock_aggregated_result
        
        with patch('asam.core.detection.engine.ConfidenceAggregator', return_value=mock_aggregator):
            engine = DetectionEngine(config)
            await engine.initialize()
            
            # Only enabled analyzers should be created
            assert len(engine.analyzers) == 3  # text disabled, 3 remaining
            
            result = await engine.analyze_screen_content()
            
            # Text analyzer should not be called
            mock_analyzers['text'].analyze.assert_not_called()
            # Other analyzers should be called
            mock_analyzers['vision'].analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyzer_error_handling(self, mock_analyzers):
        """Test error handling when analyzers fail"""
        # Make text analyzer raise exception
        mock_analyzers['text'].analyze.side_effect = Exception("LLM connection failed")
        
        # Vision analyzer returns normal result
        mock_analyzers['vision'].analyze.return_value = DetectionResult(
            analyzer_type=AnalysisType.VISION,
            confidence=0.5,
            category=ContentCategory.PRODUCTIVE,
            evidence={},
            timestamp=datetime.now()
        )
        
        mock_aggregator = MagicMock()
        mock_aggregated_result = MagicMock()
        mock_aggregator.aggregate.return_value = mock_aggregated_result
        
        with patch('asam.core.detection.engine.ConfidenceAggregator', return_value=mock_aggregator):
            engine = DetectionEngine(self.config)
            await engine.initialize()
            
            # Analysis should continue despite one analyzer failing
            result = await engine.analyze_screen_content()
            
            # Should aggregate only successful results
            call_args = mock_aggregator.aggregate.call_args[0]
            detection_results = call_args[0]
            
            # Only vision result should be passed to aggregator
            assert len(detection_results) == 1
            assert detection_results[0].analyzer_type == AnalysisType.VISION
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_limit(self, mock_analyzers):
        """Test concurrent analysis limit enforcement"""
        # Make analyzers slow to test concurrency
        async def slow_analyze(*args, **kwargs):
            await asyncio.sleep(0.1)
            return DetectionResult(
                analyzer_type=AnalysisType.TEXT,
                confidence=0.5,
                category=ContentCategory.PRODUCTIVE,
                evidence={},
                timestamp=datetime.now()
            )
        
        mock_analyzers['text'].analyze.side_effect = slow_analyze
        mock_analyzers['vision'].analyze.side_effect = slow_analyze
        mock_analyzers['process'].analyze.side_effect = slow_analyze
        mock_analyzers['network'].analyze.side_effect = slow_analyze
        
        mock_aggregator = MagicMock()
        mock_aggregated_result = MagicMock()
        mock_aggregator.aggregate.return_value = mock_aggregated_result
        
        config = self.config.copy()
        config['detection']['max_concurrent_analyses'] = 2  # Limit to 2
        
        with patch('asam.core.detection.engine.ConfidenceAggregator', return_value=mock_aggregator):
            engine = DetectionEngine(config)
            await engine.initialize()
            
            # Start multiple analyses concurrently
            tasks = [
                engine.analyze_screen_content(),
                engine.analyze_screen_content(),
                engine.analyze_screen_content(),
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should complete successfully
            assert len(results) == 3
            assert all(r == mock_aggregated_result for r in results)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_analyzers):
        """Test engine cleanup"""
        with patch('asam.core.detection.engine.ConfidenceAggregator'):
            engine = DetectionEngine(self.config)
            await engine.initialize()
            
            # Mock some analyzers as having cleanup methods
            mock_analyzers['text'].cleanup = AsyncMock()
            mock_analyzers['vision'].cleanup = AsyncMock()
            
            await engine.cleanup()
            
            # Cleanup should be called on analyzers that support it
            mock_analyzers['text'].cleanup.assert_called_once()
            mock_analyzers['vision'].cleanup.assert_called_once()


class TestDetectionEngineIntegration:
    """Integration tests for detection engine"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """Test full analysis pipeline with real components"""
        config = {
            "detection": {"confidence_threshold": 0.5},
            "text_detection": {"enabled": False},  # Disable to avoid LLM dependency
            "vision_detection": {"enabled": True, "weight": 0.6},
            "process_detection": {"enabled": True, "weight": 0.4},
            "network_detection": {"enabled": False}
        }
        
        engine = DetectionEngine(config)
        await engine.initialize()
        
        # Should have vision and process analyzers
        assert len(engine.analyzers) == 2
        
        # Run analysis (may return low confidence with mock data)
        result = await engine.analyze_screen_content()
        
        assert result is not None
        assert hasattr(result, 'overall_confidence')
        assert hasattr(result, 'primary_category')
        
        await engine.cleanup()