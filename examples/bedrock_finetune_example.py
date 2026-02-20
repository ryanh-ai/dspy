"""
Example script demonstrating how to fine-tune a model using AWS Bedrock.

This script shows how to:
1. Prepare training data
2. Configure the fine-tuning job
3. Start the fine-tuning process
4. Monitor the job status
5. Use the fine-tuned model

Note: This script requires AWS credentials with appropriate permissions.
"""

import os
import time
import dspy
from dspy.clients.utils_finetune import TrainDataFormat

# Enable experimental features (required for fine-tuning)
dspy.settings.experimental = True

def prepare_training_data():
    """Prepare sample training data for fine-tuning."""
    
    # Create a simple dataset for a Q&A task
    train_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in geography."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in geography."},
                {"role": "user", "content": "What is the capital of Germany?"},
                {"role": "assistant", "content": "The capital of Germany is Berlin."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in geography."},
                {"role": "user", "content": "What is the capital of Italy?"},
                {"role": "assistant", "content": "The capital of Italy is Rome."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in geography."},
                {"role": "user", "content": "What is the capital of Spain?"},
                {"role": "assistant", "content": "The capital of Spain is Madrid."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in geography."},
                {"role": "user", "content": "What is the capital of Portugal?"},
                {"role": "assistant", "content": "The capital of Portugal is Lisbon."}
            ]
        }
    ]
    
    return train_data

def main():
    # Check if AWS credentials are available
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        print("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables or AWS_PROFILE.")
        return
    
    # Get S3 bucket from environment variable or use a default
    s3_bucket = os.environ.get("S3_BUCKET")
    if not s3_bucket:
        print("S3_BUCKET environment variable not set. Please set it to your S3 bucket name.")
        return
    
    # Prepare S3 path
    s3_path = f"s3://{s3_bucket}/dspy-finetune/"
    
    # Get AWS profile from environment variable or use None
    aws_profile = os.environ.get("AWS_PROFILE")
    
    print("Preparing training data...")
    train_data = prepare_training_data()
    
    print(f"Creating LM with Nova Lite model...")
    lm = dspy.LM(model="bedrock/amazon.nova-lite-v1")
    
    print("Starting fine-tuning job...")
    job = lm.finetune(
        train_data=train_data,
        train_data_format=TrainDataFormat.CHAT,
        train_kwargs={
            "s3_path": s3_path,
            "aws_profile": aws_profile,
            "hyperparameters": {
                "epochCount": 3,
                "batchSize": 4,
                "learningRate": 1e-5
            },
            "start_endpoint": True  # Start an endpoint after training
        }
    )
    
    print(f"Fine-tuning job started with ID: {job.job_id}")
    print("Monitoring job status...")
    
    # Monitor job status
    while True:
        status = job.status()
        print(f"Current status: {status}")
        
        if status in [dspy.clients.utils_finetune.TrainingStatus.succeeded, 
                     dspy.clients.utils_finetune.TrainingStatus.failed,
                     dspy.clients.utils_finetune.TrainingStatus.cancelled]:
            break
        
        time.sleep(60)  # Check every minute
    
    if status == dspy.clients.utils_finetune.TrainingStatus.succeeded:
        print("Fine-tuning completed successfully!")
        
        # Get the fine-tuned model
        finetuned_model = job.result()
        print(f"Fine-tuned model ID: {finetuned_model}")
        
        # Use the fine-tuned model
        finetuned_lm = dspy.LM(model=finetuned_model)
        response = finetuned_lm("What is the capital of Japan?")
        print(f"Response from fine-tuned model: {response}")
        
        # Stop the endpoint when done
        print("Stopping the endpoint...")
        job.stop_endpoint()
    else:
        print(f"Fine-tuning failed with status: {status}")

if __name__ == "__main__":
    main()