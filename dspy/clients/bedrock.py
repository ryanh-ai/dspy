import logging
import os
import re
import time
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus, save_data, get_finetune_directory

logger = logging.getLogger(__name__)

# Fine-tunable Bedrock models
# Note: Nova models are included as requested in the feedback
BEDROCK_FINETUNABLE_MODELS = [
    # Amazon models (including Nova models as requested in the feedback)
    "amazon.nova-lite-v1",  # Nova Lite model
    "amazon.nova-micro-v1",  # Nova Micro model
    
    # Anthropic models
    "anthropic.claude-3-sonnet-20240229",
    "anthropic.claude-3-haiku-20240307",
    
    # Cohere models
    "cohere.command-r-v1",
    "cohere.command-r-plus-v1",
    
    # Meta models
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
]


class TrainingJobBedrock(TrainingJob):
    """Training job implementation for AWS Bedrock."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = None
        self.s3_data_path = None
        self.endpoint_name = None
        self.endpoint_arn = None
        self.model_arn = None
    
    def cancel(self):
        """Cancel the training job and clean up resources."""
        if self.job_id:
            try:
                import boto3
                bedrock = boto3.client('bedrock')
                bedrock.stop_model_customization_job(jobIdentifier=self.job_id)
                logger.info(f"Cancelled Bedrock fine-tuning job: {self.job_id}")
                self.job_id = None
            except Exception as e:
                logger.error(f"Failed to cancel Bedrock fine-tuning job: {e}")
        
        # Call the parent's cancel method
        super().cancel()
    
    def status(self) -> TrainingStatus:
        """Get the current status of the training job."""
        if not self.job_id:
            return TrainingStatus.not_started
        
        try:
            import boto3
            bedrock = boto3.client('bedrock')
            response = bedrock.get_model_customization_job(jobIdentifier=self.job_id)
            status = response.get('status')
            
            # Map Bedrock status to DSPy TrainingStatus
            status_mapping = {
                'InProgress': TrainingStatus.running,
                'Completed': TrainingStatus.succeeded,
                'Failed': TrainingStatus.failed,
                'Stopping': TrainingStatus.cancelled,
                'Stopped': TrainingStatus.cancelled,
                'Deleting': TrainingStatus.cancelled,
                'Creating': TrainingStatus.pending,
                'Pending': TrainingStatus.pending,
            }
            
            return status_mapping.get(status, TrainingStatus.not_started)
        except Exception as e:
            logger.error(f"Failed to get Bedrock fine-tuning job status: {e}")
            return TrainingStatus.not_started
    
    def start_endpoint(self):
        """Start the model endpoint if it exists."""
        if not self.endpoint_name:
            logger.warning("No endpoint name available to start")
            return False
        
        try:
            import boto3
            bedrock = boto3.client('bedrock')
            
            # Check if the endpoint exists
            try:
                response = bedrock.get_model_invocation_job(jobIdentifier=self.endpoint_name)
                if response.get('status') == 'InService':
                    logger.info(f"Endpoint {self.endpoint_name} is already running")
                    return True
            except:
                logger.warning(f"Endpoint {self.endpoint_name} not found")
                return False
            
            # Create the endpoint if it doesn't exist or isn't running
            if self.model_arn:
                response = bedrock.create_model_invocation_job(
                    modelId=self.model_arn,
                    jobName=self.endpoint_name,
                    provisionedModelThroughput={
                        'throughputPerModelUnit': 1
                    }
                )
                self.endpoint_arn = response.get('jobArn')
                logger.info(f"Started endpoint {self.endpoint_name}")
                return True
            else:
                logger.warning("No model ARN available to start endpoint")
                return False
        except Exception as e:
            logger.error(f"Failed to start Bedrock endpoint: {e}")
            return False
    
    def stop_endpoint(self):
        """Stop the model endpoint if it exists."""
        if not self.endpoint_name:
            logger.warning("No endpoint name available to stop")
            return False
        
        try:
            import boto3
            bedrock = boto3.client('bedrock')
            
            # Check if the endpoint exists
            try:
                response = bedrock.get_model_invocation_job(jobIdentifier=self.endpoint_name)
                if response.get('status') == 'InService':
                    bedrock.stop_model_invocation_job(jobIdentifier=self.endpoint_name)
                    logger.info(f"Stopped endpoint {self.endpoint_name}")
                    return True
                else:
                    logger.info(f"Endpoint {self.endpoint_name} is not running")
                    return True
            except:
                logger.warning(f"Endpoint {self.endpoint_name} not found")
                return False
        except Exception as e:
            logger.error(f"Failed to stop Bedrock endpoint: {e}")
            return False


class BedrockProvider(Provider):
    """Provider implementation for AWS Bedrock."""
    
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobBedrock
    
    @staticmethod
    def is_provider_model(model: str) -> bool:
        """Check if the model is a Bedrock model."""
        model = BedrockProvider._remove_provider_prefix(model)
        
        # Check if the model is a base Bedrock model
        if model in BEDROCK_FINETUNABLE_MODELS:
            return True
        
        # Check if the model is a fine-tuned Bedrock model
        # Fine-tuned models have a specific format
        match = re.match(r"custom-([^:]+)", model)
        if match:
            return True
        
        return False
    
    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        """Remove the provider prefix from the model name."""
        provider_prefix = "bedrock/"
        return model.replace(provider_prefix, "")
    
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
    def upload_data_to_s3(data_path: str, s3_path: str, aws_profile: str = None) -> str:
        """Upload data to S3 bucket.
        
        Args:
            data_path: Local path to the data file
            s3_path: S3 path in the format 's3://bucket-name/path/to/file'
            aws_profile: AWS profile to use for authentication (optional)
            
        Returns:
            S3 URI of the uploaded file
        """
        try:
            import boto3
            
            # Create a session with the specified profile if provided
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                s3 = session.client('s3')
            else:
                s3 = boto3.client('s3')
            
            # Parse the S3 path
            if not s3_path.startswith('s3://'):
                raise ValueError(f"S3 path must start with 's3://', got: {s3_path}")
            
            s3_parts = s3_path[5:].split('/', 1)
            if len(s3_parts) < 2:
                # If no key is provided, use the filename from data_path
                bucket = s3_parts[0]
                key = os.path.basename(data_path)
                s3_uri = f"s3://{bucket}/{key}"
            else:
                bucket = s3_parts[0]
                key = s3_parts[1]
                s3_uri = s3_path
                
                # If the key ends with '/', append the filename
                if key.endswith('/'):
                    key = key + os.path.basename(data_path)
                    s3_uri = f"s3://{bucket}/{key}"
            
            logger.info(f"Uploading data to {s3_uri}")
            s3.upload_file(data_path, bucket, key)
            logger.info(f"Successfully uploaded data to {s3_uri}")
            
            return s3_uri
        except Exception as e:
            logger.error(f"Failed to upload data to S3: {e}")
            raise
    
    @staticmethod
    def format_bedrock_data(train_data: List[Dict[str, Any]], data_format: TrainDataFormat) -> List[Dict[str, Any]]:
        """Format the training data for Bedrock.
        
        Args:
            train_data: List of training data dictionaries
            data_format: Format of the training data
            
        Returns:
            List of formatted training data dictionaries
        """
        formatted_data = []
        
        if data_format == TrainDataFormat.CHAT:
            for item in train_data:
                if "messages" not in item:
                    raise ValueError(f"Chat data must contain 'messages' key: {item}")
                
                # Check if there's a system message
                system_content = None
                user_messages = []
                assistant_messages = []
                
                for msg in item["messages"]:
                    if msg["role"] == "system":
                        system_content = msg["content"]
                    elif msg["role"] == "user":
                        user_messages.append(msg["content"])
                    elif msg["role"] == "assistant":
                        assistant_messages.append(msg["content"])
                
                # Ensure we have matching user/assistant pairs
                if len(user_messages) != len(assistant_messages):
                    raise ValueError(f"Unmatched user/assistant messages in: {item}")
                
                # Format for Bedrock
                bedrock_item = {
                    "input": "\n".join(user_messages),
                    "output": "\n".join(assistant_messages)
                }
                
                # Add system prompt if present
                if system_content:
                    bedrock_item["system"] = system_content
                
                formatted_data.append(bedrock_item)
        
        elif data_format == TrainDataFormat.COMPLETION:
            for item in train_data:
                if "prompt" not in item:
                    raise ValueError(f"Completion data must contain 'prompt' key: {item}")
                if "completion" not in item and "response" not in item:
                    raise ValueError(f"Completion data must contain 'completion' or 'response' key: {item}")
                
                completion = item.get("completion", item.get("response"))
                
                bedrock_item = {
                    "input": item["prompt"],
                    "output": completion
                }
                
                formatted_data.append(bedrock_item)
        
        return formatted_data
    
    @staticmethod
    def finetune(
        job: TrainingJobBedrock,
        model: str,
        train_data: List[Dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: Dict[str, Any] | None = None,
    ) -> str:
        """Fine-tune a model on AWS Bedrock.
        
        Args:
            job: Training job object
            model: Model to fine-tune
            train_data: Training data
            train_data_format: Format of the training data
            train_kwargs: Additional training parameters
            
        Returns:
            ID of the fine-tuned model
        """
        model = BedrockProvider._remove_provider_prefix(model)
        train_kwargs = train_kwargs or {}
        
        # Convert string format to enum if needed
        if isinstance(train_data_format, str):
            if train_data_format.lower() == "chat":
                train_data_format = TrainDataFormat.CHAT
            elif train_data_format.lower() == "completion":
                train_data_format = TrainDataFormat.COMPLETION
            else:
                raise ValueError(f"Unsupported data format: {train_data_format}")
        
        # Validate the data format
        BedrockProvider.validate_data_format(train_data_format)
        
        # Save the data to a local file
        logger.info("Saving the data to a file")
        data_path = save_data(train_data)
        logger.info(f"Data saved to {data_path}")
        
        # Format the data for Bedrock
        logger.info("Formatting data for Bedrock")
        bedrock_data = BedrockProvider.format_bedrock_data(train_data, train_data_format)
        
        # Save the formatted data
        bedrock_data_path = os.path.join(get_finetune_directory(), f"bedrock_{uuid.uuid4()}.jsonl")
        with open(bedrock_data_path, "w") as f:
            for item in bedrock_data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Bedrock-formatted data saved to {bedrock_data_path}")
        
        # Upload the data to S3 if s3_path is provided
        s3_path = train_kwargs.get("s3_path")
        aws_profile = train_kwargs.get("aws_profile")
        
        if s3_path:
            logger.info(f"Uploading data to S3: {s3_path}")
            s3_uri = BedrockProvider.upload_data_to_s3(bedrock_data_path, s3_path, aws_profile)
            job.s3_data_path = s3_uri
        else:
            raise ValueError("s3_path must be provided in train_kwargs")
        
        # Start the fine-tuning job
        try:
            import boto3
            
            # Create a session with the specified profile if provided
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                bedrock = session.client('bedrock')
            else:
                bedrock = boto3.client('bedrock')
            
            # Extract hyperparameters
            hyperparameters = train_kwargs.get("hyperparameters", {})
            
            # Set default hyperparameters if not provided
            if not hyperparameters:
                hyperparameters = {
                    "epochCount": 3,
                    "batchSize": 4,
                    "learningRate": 1e-5,
                }
            
            # Create a job name if not provided
            job_name = train_kwargs.get("job_name", f"dspy-finetune-{uuid.uuid4()}")
            
            # Start the fine-tuning job
            logger.info(f"Starting fine-tuning job for model {model}")
            response = bedrock.create_model_customization_job(
                customizationType="FINE_TUNING",
                baseModelIdentifier=model,
                jobName=job_name,
                hyperParameters=hyperparameters,
                trainingDataConfig={
                    "s3Uri": job.s3_data_path
                },
                outputDataConfig={
                    "s3Uri": os.path.dirname(job.s3_data_path) + "/output/"
                }
            )
            
            job.job_id = response.get("jobArn").split("/")[-1]
            logger.info(f"Fine-tuning job started with ID: {job.job_id}")
            
            # Wait for the job to complete
            BedrockProvider.wait_for_job(job)
            
            # Get the fine-tuned model ARN
            job_details = bedrock.get_model_customization_job(jobIdentifier=job.job_id)
            model_arn = job_details.get("outputModelArn")
            job.model_arn = model_arn
            
            # Create an endpoint name
            endpoint_name = f"dspy-endpoint-{uuid.uuid4()}"
            job.endpoint_name = endpoint_name
            
            # Start the endpoint if requested
            if train_kwargs.get("start_endpoint", True):
                logger.info(f"Starting endpoint for model {model_arn}")
                job.start_endpoint()
            
            # Return the model ID in the format expected by DSPy
            model_id = f"bedrock/custom-{job.job_id}"
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to start fine-tuning job: {e}")
            raise
    
    @staticmethod
    def wait_for_job(
        job: TrainingJobBedrock,
        poll_frequency: int = 60,
    ):
        """Wait for a fine-tuning job to complete.
        
        Args:
            job: Training job object
            poll_frequency: How often to check the job status (in seconds)
        """
        if not job.job_id:
            logger.warning("No job ID available to wait for")
            return
        
        try:
            import boto3
            bedrock = boto3.client('bedrock')
            
            logger.info(f"Waiting for job {job.job_id} to complete")
            done = False
            
            while not done:
                response = bedrock.get_model_customization_job(jobIdentifier=job.job_id)
                status = response.get("status")
                
                logger.info(f"Job status: {status}")
                
                if status == "Completed":
                    logger.info("Job completed successfully")
                    done = True
                elif status in ["Failed", "Stopped", "Stopping", "Deleting"]:
                    error_message = response.get("failureMessage", "Unknown error")
                    logger.error(f"Job failed: {error_message}")
                    raise Exception(f"Fine-tuning job failed: {error_message}")
                else:
                    # Calculate and display progress if available
                    metrics = response.get("metrics", {})
                    if "trainingProgress" in metrics:
                        progress = metrics["trainingProgress"]
                        logger.info(f"Training progress: {progress}%")
                    
                    # Sleep before checking again
                    time.sleep(poll_frequency)
        
        except Exception as e:
            logger.error(f"Error while waiting for job: {e}")
            raise
