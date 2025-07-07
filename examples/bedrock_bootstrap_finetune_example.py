"""
Example script demonstrating how to use BootstrapFinetune with AWS Bedrock.

This script shows how to:
1. Define a simple DSPy program
2. Create a dataset for fine-tuning
3. Configure BootstrapFinetune with Bedrock
4. Compile the program to fine-tune the model
5. Use the optimized program

Note: This script requires AWS credentials with appropriate permissions.
"""

import os
import dspy
from dspy.teleprompt import BootstrapFinetune
from dspy.primitives import Example

# Enable experimental features (required for fine-tuning)
dspy.settings.experimental = True

# Define a simple DSPy program
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)

def prepare_training_data():
    """Prepare sample training data for fine-tuning."""
    
    # Create a simple dataset for a Q&A task
    train_examples = [
        Example(
            question="What is the capital of France?",
            answer="The capital of France is Paris."
        ),
        Example(
            question="What is the capital of Germany?",
            answer="The capital of Germany is Berlin."
        ),
        Example(
            question="What is the capital of Italy?",
            answer="The capital of Italy is Rome."
        ),
        Example(
            question="What is the capital of Spain?",
            answer="The capital of Spain is Madrid."
        ),
        Example(
            question="What is the capital of Portugal?",
            answer="The capital of Portugal is Lisbon."
        )
    ]
    
    return train_examples

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
    s3_path = f"s3://{s3_bucket}/dspy-bootstrap-finetune/"
    
    # Get AWS profile from environment variable or use None
    aws_profile = os.environ.get("AWS_PROFILE")
    
    print("Preparing training data...")
    train_examples = prepare_training_data()
    
    print("Creating base LM with Nova Lite model...")
    base_lm = dspy.LM(model="bedrock/amazon.nova-lite-v1")
    
    print("Creating program...")
    program = SimpleQA()
    
    # Attach the LM to the program
    for predictor in program.predictors():
        predictor.lm = base_lm
    
    print("Creating BootstrapFinetune optimizer...")
    optimizer = BootstrapFinetune(
        train_kwargs={
            "s3_path": s3_path,
            "aws_profile": aws_profile,
            "hyperparameters": {
                "epochCount": 3,
                "batchSize": 4,
                "learningRate": 1e-5
            },
            "start_endpoint": True
        }
    )
    
    print("Compiling program (this will start the fine-tuning process)...")
    optimized_program = optimizer.compile(
        student=program,
        trainset=train_examples
    )
    
    print("Fine-tuning completed!")
    
    # Use the optimized program
    print("Testing the optimized program...")
    result = optimized_program(question="What is the capital of Japan?")
    print(f"Answer: {result.answer}")
    
    # Get the fine-tuned LM
    finetuned_lm = optimized_program.generate_answer.lm
    
    # Stop the endpoint when done
    print("Stopping the endpoint...")
    # We need to access the job through the LM's provider
    # This is a bit hacky but works for demonstration purposes
    job_id = finetuned_lm.model.split("custom-")[1]
    job = dspy.clients.bedrock.TrainingJobBedrock()
    job.job_id = job_id
    job.endpoint_name = f"dspy-endpoint-{job_id}"
    job.stop_endpoint()

if __name__ == "__main__":
    main()