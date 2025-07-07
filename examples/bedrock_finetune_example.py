"""
Example of fine-tuning a model using the Bedrock provider.

Prerequisites:
1. Install the required packages:
   pip install dspy boto3

2. Set up AWS credentials:
   - Configure AWS CLI with `aws configure`
   - Or set environment variables:
     export AWS_ACCESS_KEY_ID=your_access_key
     export AWS_SECRET_ACCESS_KEY=your_secret_key
     export AWS_REGION=your_region

3. Set up S3 bucket and IAM role:
   - Create an S3 bucket for training data
   - Create an IAM role with permissions for Bedrock and S3
   - Set environment variables:
     export DSPY_BEDROCK_S3_BUCKET=your_bucket_name
     export DSPY_BEDROCK_ROLE_ARN=your_role_arn
"""

import os
import dspy
from dspy.clients.bedrock import BedrockProvider
from dspy.clients.utils_finetune import TrainDataFormat

# Create a simple classification module
class SimpleClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text -> category")
        
    def forward(self, text):
        return self.classify(text=text)

def main():
    # Check if boto3 is installed
    try:
        import boto3
    except ImportError:
        print("boto3 is required for this example. Install it with `pip install boto3`.")
        return
        
    # Check if AWS credentials are set
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        print("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        return
        
    # Check if S3 bucket and IAM role are set
    if not os.environ.get("DSPY_BEDROCK_S3_BUCKET"):
        print("S3 bucket not set. Please set DSPY_BEDROCK_S3_BUCKET environment variable.")
        return
        
    if not os.environ.get("DSPY_BEDROCK_ROLE_ARN"):
        print("IAM role not set. Please set DSPY_BEDROCK_ROLE_ARN environment variable.")
        return
    
    # Create a Bedrock LM
    lm = dspy.LM(
        model="bedrock/anthropic.claude-3-sonnet-20240229",
        provider=BedrockProvider()
    )
    
    # Set the LM as the default
    dspy.settings.configure(lm=lm)
    
    # Create training data
    train_data = [
        {
            "messages": [
                {"role": "user", "content": "Classify this text: 'I love this product!'"},
                {"role": "assistant", "content": "category: positive"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Classify this text: 'This is terrible.'"},
                {"role": "assistant", "content": "category: negative"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Classify this text: 'It's okay I guess.'"},
                {"role": "assistant", "content": "category: neutral"}
            ]
        },
        # Add more examples as needed
    ]
    
    # Start fine-tuning
    print("Starting fine-tuning...")
    job = lm.finetune(
        train_data=train_data,
        train_data_format=TrainDataFormat.CHAT,
        train_kwargs={
            "epochCount": 3,
            "batchSize": 4,
            "learningRate": 1e-5,
            # Optional: specify S3 path directly
            # "s3_path": "s3://your-bucket/your-prefix/",
        }
    )
    
    # Wait for the job to complete (this is already done in the finetune method)
    print(f"Fine-tuning job status: {job.status()}")
    
    if job.status() == "succeeded":
        print(f"Fine-tuned model ARN: {job.model_arn}")
        
        # Start an endpoint for the fine-tuned model
        print("Starting endpoint...")
        endpoint_name = job.start_endpoint()
        print(f"Endpoint started: {endpoint_name}")
        
        # Use the fine-tuned model
        # Note: In a real application, you would create a new LM with the fine-tuned model
        
        # Stop the endpoint when done
        print("Stopping endpoint...")
        job.stop_endpoint()
        print("Endpoint stopped")
    else:
        print(f"Fine-tuning failed with status: {job.status()}")

if __name__ == "__main__":
    main()