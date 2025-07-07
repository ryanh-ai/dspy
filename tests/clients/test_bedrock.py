import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

import dspy
from dspy.clients.bedrock import BedrockProvider, TrainingJobBedrock
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus


class TestBedrockProvider(unittest.TestCase):
    """Test the Bedrock provider."""

    def setUp(self):
        """Set up the test environment."""
        # Skip tests if boto3 is not installed
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 not installed")
            
        # Create a mock provider
        self.provider = BedrockProvider()
        
    def test_is_provider_model(self):
        """Test the is_provider_model method."""
        # Test with a valid model
        self.assertTrue(BedrockProvider.is_provider_model("bedrock/anthropic.claude-3-sonnet-20240229"))
        
        # Test with an invalid model
        self.assertFalse(BedrockProvider.is_provider_model("openai/gpt-4"))
        
    def test_get_model_family(self):
        """Test the _get_model_family method."""
        self.assertEqual(BedrockProvider._get_model_family("anthropic.claude-3-sonnet-20240229"), "anthropic")
        self.assertEqual(BedrockProvider._get_model_family("meta.llama3-8b"), "meta")
        self.assertEqual(BedrockProvider._get_model_family("amazon.titan-text-express-v1"), "amazon")
        self.assertEqual(BedrockProvider._get_model_family("cohere.command-r-v1"), "cohere")
        self.assertEqual(BedrockProvider._get_model_family("unknown-model"), "unknown")
        
    def test_get_default_hyperparameters(self):
        """Test the _get_default_hyperparameters method."""
        # Test with a valid model family
        hyperparams = BedrockProvider._get_default_hyperparameters("anthropic.claude-3-sonnet-20240229")
        self.assertIn("epochCount", hyperparams)
        self.assertIn("batchSize", hyperparams)
        self.assertIn("learningRate", hyperparams)
        
        # Test with an unknown model family
        hyperparams = BedrockProvider._get_default_hyperparameters("unknown-model")
        self.assertEqual(hyperparams, {})
        
    @patch("boto3.client")
    def test_training_job_status(self, mock_boto3_client):
        """Test the status method of TrainingJobBedrock."""
        # Create a mock job
        job = TrainingJobBedrock()
        job.job_id = "test-job-id"
        
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        # Test each status
        for bedrock_status, expected_status in [
            ("InProgress", TrainingStatus.running),
            ("Completed", TrainingStatus.succeeded),
            ("Failed", TrainingStatus.failed),
            ("Stopping", TrainingStatus.running),
            ("Stopped", TrainingStatus.cancelled),
        ]:
            mock_client.get_model_customization_job.return_value = {"status": bedrock_status}
            self.assertEqual(job.status(), expected_status)
            
        # Test with no job ID
        job.job_id = None
        self.assertEqual(job.status(), TrainingStatus.not_started)
        
    @patch("boto3.client")
    def test_convert_to_bedrock_format(self, mock_boto3_client):
        """Test the _convert_to_bedrock_format method."""
        # Test data
        chat_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ]
        
        # Test conversion for different model families
        anthropic_data = BedrockProvider._convert_to_bedrock_format(
            chat_data, TrainDataFormat.CHAT, "anthropic.claude-3-sonnet-20240229"
        )
        self.assertEqual(len(anthropic_data), 1)
        self.assertIn("messages", anthropic_data[0])
        
        meta_data = BedrockProvider._convert_to_bedrock_format(
            chat_data, TrainDataFormat.CHAT, "meta.llama3-8b"
        )
        self.assertEqual(len(meta_data), 1)
        self.assertIn("conversations", meta_data[0])
        
        amazon_data = BedrockProvider._convert_to_bedrock_format(
            chat_data, TrainDataFormat.CHAT, "amazon.titan-text-express-v1"
        )
        self.assertEqual(len(amazon_data), 1)
        self.assertIn("inputText", amazon_data[0])
        self.assertIn("outputText", amazon_data[0])
        
        cohere_data = BedrockProvider._convert_to_bedrock_format(
            chat_data, TrainDataFormat.CHAT, "cohere.command-r-v1"
        )
        self.assertEqual(len(cohere_data), 1)
        self.assertIn("prompt", cohere_data[0])
        self.assertIn("completion", cohere_data[0])


if __name__ == "__main__":
    unittest.main()