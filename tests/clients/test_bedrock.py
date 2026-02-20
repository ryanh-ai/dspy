"""Test the AWS Bedrock finetuning and deployment.

This test requires valid AWS credentials, so it is skipped by default. It can be run manually
when AWS credentials are properly configured.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import dspy
from dspy.clients.bedrock import BedrockProvider, TrainingJobBedrock
from dspy.clients.utils_finetune import TrainDataFormat

# Skip all tests by default since they require AWS credentials
pytestmark = pytest.mark.skip(reason="AWS credentials required")


def test_validate_data_format():
    """Test that the validate_data_format method works correctly."""
    # Valid formats should not raise an exception
    BedrockProvider.validate_data_format(TrainDataFormat.CHAT)
    BedrockProvider.validate_data_format(TrainDataFormat.COMPLETION)
    
    # Invalid format should raise an exception
    with pytest.raises(ValueError):
        BedrockProvider.validate_data_format("invalid_format")


def test_format_bedrock_data_chat():
    """Test formatting chat data for Bedrock."""
    chat_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ]
        }
    ]
    
    formatted_data = BedrockProvider.format_bedrock_data(chat_data, TrainDataFormat.CHAT)
    
    assert len(formatted_data) == 1
    assert formatted_data[0]["input"] == "Hello, how are you?"
    assert formatted_data[0]["output"] == "I'm doing great, thank you!"
    assert formatted_data[0]["system"] == "You are a helpful assistant."


def test_format_bedrock_data_completion():
    """Test formatting completion data for Bedrock."""
    completion_data = [
        {
            "prompt": "What is the capital of France?",
            "completion": "Paris is the capital of France."
        }
    ]
    
    formatted_data = BedrockProvider.format_bedrock_data(completion_data, TrainDataFormat.COMPLETION)
    
    assert len(formatted_data) == 1
    assert formatted_data[0]["input"] == "What is the capital of France?"
    assert formatted_data[0]["output"] == "Paris is the capital of France."


@patch('boto3.client')
def test_upload_data_to_s3(mock_boto_client):
    """Test uploading data to S3."""
    # Create a mock S3 client
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3
    
    # Create a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test data")
        tmp_path = tmp.name
    
    try:
        # Test with full S3 path
        s3_path = "s3://my-bucket/my-folder/my-file.jsonl"
        result = BedrockProvider.upload_data_to_s3(tmp_path, s3_path)
        
        # Check that the S3 client was called correctly
        mock_s3.upload_file.assert_called_once_with(tmp_path, "my-bucket", "my-folder/my-file.jsonl")
        assert result == s3_path
        
        # Reset mock
        mock_s3.reset_mock()
        
        # Test with bucket only
        s3_path = "s3://my-bucket/"
        result = BedrockProvider.upload_data_to_s3(tmp_path, s3_path)
        
        # Check that the S3 client was called correctly with the filename from tmp_path
        mock_s3.upload_file.assert_called_once_with(tmp_path, "my-bucket", os.path.basename(tmp_path))
        assert result == f"s3://my-bucket/{os.path.basename(tmp_path)}"
    finally:
        # Clean up
        os.unlink(tmp_path)


@patch('boto3.client')
def test_training_job_status(mock_boto_client):
    """Test getting the status of a training job."""
    # Create a mock Bedrock client
    mock_bedrock = MagicMock()
    mock_boto_client.return_value = mock_bedrock
    
    # Set up the mock response
    mock_bedrock.get_model_customization_job.return_value = {"status": "InProgress"}
    
    # Create a training job
    job = TrainingJobBedrock()
    job.job_id = "test-job-id"
    
    # Check the status
    status = job.status()
    
    # Verify the result
    assert status == dspy.clients.utils_finetune.TrainingStatus.running
    mock_bedrock.get_model_customization_job.assert_called_once_with(jobIdentifier="test-job-id")


@patch('boto3.client')
def test_start_endpoint(mock_boto_client):
    """Test starting an endpoint."""
    # Create a mock Bedrock client
    mock_bedrock = MagicMock()
    mock_boto_client.return_value = mock_bedrock
    
    # Set up the mock response for get_model_invocation_job
    mock_bedrock.get_model_invocation_job.side_effect = Exception("Not found")
    
    # Set up the mock response for create_model_invocation_job
    mock_bedrock.create_model_invocation_job.return_value = {"jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-endpoint"}
    
    # Create a training job
    job = TrainingJobBedrock()
    job.endpoint_name = "test-endpoint"
    job.model_arn = "arn:aws:bedrock:us-west-2:123456789012:model/custom-model-id"
    
    # Start the endpoint
    result = job.start_endpoint()
    
    # Verify the result
    assert result is True
    mock_bedrock.create_model_invocation_job.assert_called_once_with(
        modelId=job.model_arn,
        jobName=job.endpoint_name,
        provisionedModelThroughput={"throughputPerModelUnit": 1}
    )
    assert job.endpoint_arn == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-endpoint"


@patch('boto3.client')
def test_stop_endpoint(mock_boto_client):
    """Test stopping an endpoint."""
    # Create a mock Bedrock client
    mock_bedrock = MagicMock()
    mock_boto_client.return_value = mock_bedrock
    
    # Set up the mock response
    mock_bedrock.get_model_invocation_job.return_value = {"status": "InService"}
    
    # Create a training job
    job = TrainingJobBedrock()
    job.endpoint_name = "test-endpoint"
    
    # Stop the endpoint
    result = job.stop_endpoint()
    
    # Verify the result
    assert result is True
    mock_bedrock.stop_model_invocation_job.assert_called_once_with(jobIdentifier="test-endpoint")


@patch('dspy.clients.bedrock.BedrockProvider.upload_data_to_s3')
@patch('dspy.clients.bedrock.BedrockProvider.wait_for_job')
@patch('boto3.client')
def test_finetune(mock_boto_client, mock_wait_for_job, mock_upload_to_s3):
    """Test the finetune method."""
    # Create a mock Bedrock client
    mock_bedrock = MagicMock()
    mock_boto_client.return_value = mock_bedrock
    
    # Set up the mock responses
    mock_upload_to_s3.return_value = "s3://my-bucket/my-data.jsonl"
    mock_bedrock.create_model_customization_job.return_value = {
        "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-customization-job/test-job-id"
    }
    mock_bedrock.get_model_customization_job.return_value = {
        "outputModelArn": "arn:aws:bedrock:us-west-2:123456789012:model/custom-model-id"
    }
    
    # Create test data
    train_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thank you!"},
            ]
        }
    ]
    
    # Create a training job
    job = TrainingJobBedrock()
    
    # Call finetune
    model_id = BedrockProvider.finetune(
        job=job,
        model="amazon.nova-lite-v1",
        train_data=train_data,
        train_data_format=TrainDataFormat.CHAT,
        train_kwargs={
            "s3_path": "s3://my-bucket/",
            "hyperparameters": {
                "epochCount": 5,
                "batchSize": 8,
                "learningRate": 2e-5,
            },
            "start_endpoint": False
        }
    )
    
    # Verify the result
    assert model_id == "bedrock/custom-test-job-id"
    assert job.job_id == "test-job-id"
    assert job.model_arn == "arn:aws:bedrock:us-west-2:123456789012:model/custom-model-id"
    
    # Verify the API calls
    mock_bedrock.create_model_customization_job.assert_called_once()
    args = mock_bedrock.create_model_customization_job.call_args[1]
    assert args["customizationType"] == "FINE_TUNING"
    assert args["baseModelIdentifier"] == "amazon.nova-lite-v1"
    assert args["hyperParameters"] == {
        "epochCount": 5,
        "batchSize": 8,
        "learningRate": 2e-5,
    }
    assert args["trainingDataConfig"]["s3Uri"] == "s3://my-bucket/my-data.jsonl"