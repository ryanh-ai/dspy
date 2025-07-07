import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import json
import uuid

try:
    import boto3
    import botocore
except ImportError:
    boto3 = None
    botocore = None

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import (
    TrainDataFormat, 
    TrainingStatus, 
    save_data
)

logger = logging.getLogger(__name__)

# List of models supported by Amazon Bedrock for fine-tuning
# This list should be updated as Amazon adds more models
_BEDROCK_FINETUNABLE_MODELS = [
    "anthropic.claude-3-sonnet-20240229",
    "anthropic.claude-3-haiku-20240307",
    "meta.llama2-13b",
    "meta.llama2-70b",
    "meta.llama3-8b",
    "meta.llama3-70b",
    "amazon.titan-text-express-v1",
    "amazon.titan-text-premier-v1",
    "cohere.command-r-v1",
    "cohere.command-r-plus-v1",
]

# Default hyperparameters for different model families
DEFAULT_HYPERPARAMETERS = {
    "anthropic": {
        "epochCount": 3,
        "batchSize": 4,
        "learningRate": 1e-5,
    },
    "meta": {
        "epochCount": 3,
        "batchSize": 1,
        "learningRate": 1e-5,
    },
    "amazon": {
        "epochCount": 5,
        "batchSize": 1,
        "learningRate": 2e-5,
    },
    "cohere": {
        "epochCount": 3,
        "batchSize": 2,
        "learningRate": 1e-5,
    }
}


class TrainingJobBedrock(TrainingJob):
    """Training job implementation for Amazon Bedrock."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = None
        self.s3_data_path = None
        self.s3_bucket = None
        self.s3_key = None
        self.model_arn = None
        self.endpoint_name = None
        
    def cancel(self):
        """Cancel the training job and clean up resources."""
        if self.job_id:
            try:
                bedrock_client = boto3.client('bedrock')
                bedrock_client.stop_model_customization_job(jobIdentifier=self.job_id)
                logger.info(f"Cancelled Bedrock fine-tuning job: {self.job_id}")
            except Exception as e:
                logger.error(f"Failed to cancel Bedrock job {self.job_id}: {str(e)}")
        
        # Call the parent's cancel method
        super().cancel()
        
    def status(self) -> TrainingStatus:
        """Get the current status of the training job."""
        if not self.job_id:
            return TrainingStatus.not_started
            
        try:
            bedrock_client = boto3.client('bedrock')
            response = bedrock_client.get_model_customization_job(jobIdentifier=self.job_id)
            job_status = response.get('status')
            
            # Map Bedrock status to DSPy TrainingStatus
            status_mapping = {
                'InProgress': TrainingStatus.running,
                'Completed': TrainingStatus.succeeded,
                'Failed': TrainingStatus.failed,
                'Stopping': TrainingStatus.running,
                'Stopped': TrainingStatus.cancelled,
                'Deleting': TrainingStatus.cancelled,
                'Creating': TrainingStatus.pending,
                'Pending': TrainingStatus.pending,
            }
            
            return status_mapping.get(job_status, TrainingStatus.running)
        except Exception as e:
            logger.error(f"Failed to get status for job {self.job_id}: {str(e)}")
            return TrainingStatus.failed
            
    def start_endpoint(self):
        """Start a model deployment endpoint for the fine-tuned model."""
        if not self.model_arn:
            raise ValueError("No fine-tuned model available to deploy")
            
        try:
            bedrock_client = boto3.client('bedrock')
            
            # Generate a unique endpoint name if not provided
            if not self.endpoint_name:
                self.endpoint_name = f"dspy-bedrock-endpoint-{uuid.uuid4().hex[:8]}"
                
            # Create model invocation endpoint
            response = bedrock_client.create_model_invocation_job(
                modelId=self.model_arn,
                endpointName=self.endpoint_name,
                executionRoleArn=self.train_kwargs.get('executionRoleArn'),
            )
            
            logger.info(f"Started endpoint deployment: {self.endpoint_name}")
            return self.endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to start endpoint: {str(e)}")
            raise
            
    def stop_endpoint(self):
        """Stop and delete the model deployment endpoint."""
        if not self.endpoint_name:
            logger.warning("No endpoint to stop")
            return
            
        try:
            bedrock_client = boto3.client('bedrock')
            bedrock_client.delete_model_invocation_job(
                jobIdentifier=self.endpoint_name
            )
            logger.info(f"Stopped endpoint: {self.endpoint_name}")
            self.endpoint_name = None
        except Exception as e:
            logger.error(f"Failed to stop endpoint: {str(e)}")
            raise


class BedrockProvider(Provider):
    """Provider implementation for Amazon Bedrock fine-tuning."""
    
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobBedrock
        
        # Check if boto3 is installed
        if boto3 is None:
            logger.warning(
                "The boto3 package is required to use the BedrockProvider. "
                "Please install it with `pip install boto3`."
            )
            self.finetunable = False
    
    @staticmethod
    def is_provider_model(model: str) -> bool:
        """Check if the model is supported by this provider."""
        model = BedrockProvider._remove_provider_prefix(model)
        
        # Check if the model is in the list of supported models
        if model in _BEDROCK_FINETUNABLE_MODELS:
            return True
            
        return False
    
    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        """Remove the provider prefix from the model name."""
        provider_prefix = "bedrock/"
        return model.replace(provider_prefix, "")
    
    @staticmethod
    def _get_model_family(model: str) -> str:
        """Get the model family from the model name."""
        if model.startswith("anthropic."):
            return "anthropic"
        elif model.startswith("meta."):
            return "meta"
        elif model.startswith("amazon."):
            return "amazon"
        elif model.startswith("cohere."):
            return "cohere"
        else:
            return "unknown"
    
    @staticmethod
    def _convert_to_bedrock_format(
        train_data: List[Dict[str, Any]], 
        train_data_format: TrainDataFormat,
        model: str
    ) -> List[Dict[str, Any]]:
        """Convert training data to the format expected by Bedrock."""
        model_family = BedrockProvider._get_model_family(model)
        
        if train_data_format == TrainDataFormat.CHAT:
            if model_family == "anthropic":
                # Convert to Anthropic format
                bedrock_data = []
                for item in train_data:
                    messages = item.get("messages", [])
                    if not messages:
                        continue
                        
                    # Format for Claude models
                    formatted_item = {
                        "messages": []
                    }
                    
                    for msg in messages:
                        formatted_item["messages"].append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    bedrock_data.append(formatted_item)
                return bedrock_data
                
            elif model_family == "meta":
                # Convert to Llama format
                bedrock_data = []
                for item in train_data:
                    messages = item.get("messages", [])
                    if not messages:
                        continue
                        
                    # Format for Llama models
                    formatted_item = {
                        "conversations": []
                    }
                    
                    for msg in messages:
                        formatted_item["conversations"].append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    
                    bedrock_data.append(formatted_item)
                return bedrock_data
                
            elif model_family == "amazon":
                # Convert to Titan format
                bedrock_data = []
                for item in train_data:
                    messages = item.get("messages", [])
                    if not messages:
                        continue
                        
                    # Format for Titan models
                    formatted_item = {
                        "inputText": "",
                        "outputText": ""
                    }
                    
                    # Extract user and assistant messages
                    for msg in messages:
                        if msg["role"] == "user":
                            formatted_item["inputText"] += msg["content"] + "\n"
                        elif msg["role"] == "assistant":
                            formatted_item["outputText"] += msg["content"] + "\n"
                    
                    # Clean up trailing newlines
                    formatted_item["inputText"] = formatted_item["inputText"].strip()
                    formatted_item["outputText"] = formatted_item["outputText"].strip()
                    
                    bedrock_data.append(formatted_item)
                return bedrock_data
                
            elif model_family == "cohere":
                # Convert to Cohere format
                bedrock_data = []
                for item in train_data:
                    messages = item.get("messages", [])
                    if not messages:
                        continue
                        
                    # Format for Cohere models
                    formatted_item = {
                        "prompt": "",
                        "completion": ""
                    }
                    
                    # Extract user and assistant messages
                    for msg in messages:
                        if msg["role"] == "user":
                            formatted_item["prompt"] += msg["content"] + "\n"
                        elif msg["role"] == "assistant":
                            formatted_item["completion"] += msg["content"] + "\n"
                    
                    # Clean up trailing newlines
                    formatted_item["prompt"] = formatted_item["prompt"].strip()
                    formatted_item["completion"] = formatted_item["completion"].strip()
                    
                    bedrock_data.append(formatted_item)
                return bedrock_data
        
        # If we don't have a specific conversion, return the original data
        logger.warning(f"No specific conversion for {model_family} with format {train_data_format}. Using original data.")
        return train_data
        
    @staticmethod
    def upload_to_s3(
        data_path: str, 
        s3_path: str = None, 
        bucket: str = None, 
        key: str = None
    ) -> Dict[str, str]:
        """
        Upload training data to S3.
        
        Args:
            data_path: Local path to the data file
            s3_path: Optional S3 URI (s3://bucket/key)
            bucket: Optional S3 bucket name (used if s3_path not provided)
            key: Optional S3 key (used if s3_path not provided)
            
        Returns:
            Dictionary with 'bucket' and 'key' values
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Parse S3 path if provided
        if s3_path:
            if not s3_path.startswith("s3://"):
                raise ValueError("S3 path must start with 's3://'")
                
            parts = s3_path[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError("Invalid S3 path format. Expected: s3://bucket/key")
                
            bucket = parts[0]
            key = parts[1]
        
        # Generate bucket/key if not provided
        if not bucket:
            bucket = os.environ.get("DSPY_BEDROCK_S3_BUCKET")
            if not bucket:
                raise ValueError(
                    "S3 bucket not provided. Either specify a bucket, an s3_path, "
                    "or set the DSPY_BEDROCK_S3_BUCKET environment variable."
                )
        
        if not key:
            # Generate a unique key based on timestamp and file name
            file_name = os.path.basename(data_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            key = f"dspy-finetune/{timestamp}-{file_name}"
        
        try:
            # Upload the file to S3
            s3_client = boto3.client('s3')
            s3_client.upload_file(data_path, bucket, key)
            logger.info(f"Uploaded data to s3://{bucket}/{key}")
            
            return {"bucket": bucket, "key": key}
        except Exception as e:
            logger.error(f"Failed to upload data to S3: {str(e)}")
            raise
    
    @staticmethod
    def validate_data_format(data_format: TrainDataFormat):
        """Validate that the data format is supported by Bedrock."""
        supported_data_formats = [
            TrainDataFormat.CHAT,
            TrainDataFormat.COMPLETION,
        ]
        if data_format not in supported_data_formats:
            err_msg = f"Bedrock does not support the data format {data_format}."
            raise ValueError(err_msg)
    
    @staticmethod
    def _get_default_hyperparameters(model: str) -> Dict[str, Any]:
        """Get default hyperparameters for the model."""
        model_family = BedrockProvider._get_model_family(model)
        return DEFAULT_HYPERPARAMETERS.get(model_family, {})
    
    @staticmethod
    def _prepare_hyperparameters(model: str, train_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare hyperparameters for the fine-tuning job."""
        # Get default hyperparameters for the model family
        hyperparameters = BedrockProvider._get_default_hyperparameters(model)
        
        # Update with user-provided hyperparameters
        if train_kwargs:
            for key, value in train_kwargs.items():
                if key in ["epochCount", "batchSize", "learningRate"]:
                    hyperparameters[key] = value
        
        return hyperparameters
    
    @staticmethod
    def finetune(
        job: TrainingJobBedrock,
        model: str,
        train_data: List[Dict[str, Any]],
        train_data_format: TrainDataFormat | str,
        train_kwargs: Dict[str, Any] | None = None,
    ) -> str:
        """
        Start a fine-tuning job on Amazon Bedrock.
        
        Args:
            job: The training job object
            model: The model to fine-tune
            train_data: The training data
            train_data_format: The format of the training data
            train_kwargs: Additional training parameters
            
        Returns:
            The ARN of the fine-tuned model
        """
        model = BedrockProvider._remove_provider_prefix(model)
        train_kwargs = train_kwargs or {}
        
        logger.info("[Bedrock Provider] Validating the data format")
        BedrockProvider.validate_data_format(train_data_format)
        
        logger.info("[Bedrock Provider] Converting data to Bedrock format")
        bedrock_data = BedrockProvider._convert_to_bedrock_format(
            train_data, 
            train_data_format,
            model
        )
        
        logger.info("[Bedrock Provider] Saving the data to a file")
        data_path = save_data(bedrock_data)
        logger.info(f"[Bedrock Provider] Data saved to {data_path}")
        
        # Check if S3 path is provided in train_kwargs
        s3_path = train_kwargs.get("s3_path")
        
        logger.info("[Bedrock Provider] Uploading the data to S3")
        s3_info = BedrockProvider.upload_to_s3(
            data_path=data_path,
            s3_path=s3_path,
            bucket=train_kwargs.get("s3_bucket"),
            key=train_kwargs.get("s3_key")
        )
        
        job.s3_bucket = s3_info["bucket"]
        job.s3_key = s3_info["key"]
        job.s3_data_path = f"s3://{job.s3_bucket}/{job.s3_key}"
        
        # Prepare hyperparameters
        hyperparameters = BedrockProvider._prepare_hyperparameters(model, train_kwargs)
        
        # Get required role ARN
        execution_role_arn = train_kwargs.get("executionRoleArn")
        if not execution_role_arn:
            execution_role_arn = os.environ.get("DSPY_BEDROCK_ROLE_ARN")
            if not execution_role_arn:
                raise ValueError(
                    "executionRoleArn not provided. Either specify it in train_kwargs "
                    "or set the DSPY_BEDROCK_ROLE_ARN environment variable."
                )
        
        # Generate a unique job name
        job_name = train_kwargs.get("jobName", f"dspy-finetune-{uuid.uuid4().hex[:8]}")
        
        logger.info("[Bedrock Provider] Starting fine-tuning job")
        try:
            bedrock_client = boto3.client('bedrock')
            response = bedrock_client.create_model_customization_job(
                jobName=job_name,
                customizationType="FINE_TUNING",
                baseModelIdentifier=model,
                trainingDataConfig={
                    "s3Uri": job.s3_data_path
                },
                hyperParameters=hyperparameters,
                roleArn=execution_role_arn,
                outputDataConfig={
                    "s3Uri": f"s3://{job.s3_bucket}/dspy-finetune-output/"
                }
            )
            
            job.job_id = response["jobArn"]
            logger.info(f"[Bedrock Provider] Job started with ID: {job.job_id}")
            
            # Wait for the job to complete
            logger.info("[Bedrock Provider] Waiting for training to complete")
            BedrockProvider.wait_for_job(job)
            
            # Get the fine-tuned model ARN
            if job.status() == TrainingStatus.succeeded:
                job_details = bedrock_client.get_model_customization_job(jobIdentifier=job.job_id)
                job.model_arn = job_details.get("outputModelArn")
                logger.info(f"[Bedrock Provider] Fine-tuned model ARN: {job.model_arn}")
                return job.model_arn
            else:
                status = job.status()
                raise Exception(f"Fine-tuning job failed with status: {status}")
                
        except Exception as e:
            logger.error(f"[Bedrock Provider] Fine-tuning failed: {str(e)}")
            raise
    
    @staticmethod
    def wait_for_job(
        job: TrainingJobBedrock,
        poll_frequency: int = 60,
    ):
        """Wait for the fine-tuning job to complete."""
        done = False
        while not done:
            status = job.status()
            logger.info(f"[Bedrock Provider] Job status: {status}")
            
            if BedrockProvider.is_terminal_training_status(status):
                done = True
            else:
                time.sleep(poll_frequency)
    
    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        """Check if the training status is terminal."""
        return status in [
            TrainingStatus.succeeded,
            TrainingStatus.failed,
            TrainingStatus.cancelled,
        ]