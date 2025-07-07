# Fine-tuning with DSPy

DSPy provides a unified interface for fine-tuning language models across different providers. This guide explains how to use DSPy's fine-tuning capabilities.

## Supported Providers

DSPy currently supports fine-tuning with the following providers:

- **OpenAI**: Fine-tune OpenAI models like GPT-3.5
- **Bedrock**: Fine-tune Amazon Bedrock models like Claude, Llama, Titan, and Cohere

## Basic Fine-tuning

Here's a simple example of fine-tuning an OpenAI model:

```python
import dspy
from dspy.clients.utils_finetune import TrainDataFormat

# Create an LM with OpenAI
lm = dspy.LM("openai/gpt-3.5-turbo")

# Prepare training data
train_data = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    },
    # Add more examples...
]

# Start fine-tuning
job = lm.finetune(
    train_data=train_data,
    train_data_format=TrainDataFormat.CHAT
)

# Wait for the job to complete (this happens automatically in finetune)
print(f"Fine-tuned model: {job.model}")
```

## Fine-tuning with Amazon Bedrock

DSPy also supports fine-tuning models on Amazon Bedrock. This includes models from Anthropic (Claude), Meta (Llama), Amazon (Titan), and Cohere.

### Prerequisites

1. Install the required packages:
   ```
   pip install dspy boto3
   ```

2. Set up AWS credentials:
   - Configure AWS CLI with `aws configure`
   - Or set environment variables:
     ```
     export AWS_ACCESS_KEY_ID=your_access_key
     export AWS_SECRET_ACCESS_KEY=your_secret_key
     export AWS_REGION=your_region
     ```

3. Set up S3 bucket and IAM role:
   - Create an S3 bucket for training data
   - Create an IAM role with permissions for Bedrock and S3
   - Set environment variables:
     ```
     export DSPY_BEDROCK_S3_BUCKET=your_bucket_name
     export DSPY_BEDROCK_ROLE_ARN=your_role_arn
     ```

### Example

```python
import dspy
from dspy.clients.bedrock import BedrockProvider
from dspy.clients.utils_finetune import TrainDataFormat

# Create a Bedrock LM
lm = dspy.LM(
    model="bedrock/anthropic.claude-3-sonnet-20240229",
    provider=BedrockProvider()
)

# Prepare training data
train_data = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    },
    # Add more examples...
]

# Start fine-tuning
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

# Check the status
print(f"Fine-tuning job status: {job.status()}")

# If successful, deploy an endpoint
if job.status() == "succeeded":
    endpoint_name = job.start_endpoint()
    print(f"Endpoint deployed: {endpoint_name}")
    
    # When done, stop the endpoint
    job.stop_endpoint()
```

### Supported Hyperparameters

The Bedrock provider supports the following hyperparameters:

- `epochCount`: Number of training epochs
- `batchSize`: Batch size for training
- `learningRate`: Learning rate for training

Different model families may have different default values and constraints for these hyperparameters.

### Model Format Conversion

The Bedrock provider automatically converts your training data to the format expected by each model family:

- **Anthropic Claude**: Uses the standard chat format with messages
- **Meta Llama**: Uses a conversations format
- **Amazon Titan**: Converts to inputText/outputText format
- **Cohere**: Converts to prompt/completion format

## Advanced Options

### Custom S3 Path

You can specify a custom S3 path for your training data:

```python
job = lm.finetune(
    train_data=train_data,
    train_data_format=TrainDataFormat.CHAT,
    train_kwargs={
        "s3_path": "s3://your-bucket/your-prefix/",
    }
)
```

### Endpoint Management

You can start and stop endpoints for your fine-tuned models:

```python
# Start an endpoint
endpoint_name = job.start_endpoint()

# Stop the endpoint when done
job.stop_endpoint()
```

## Data Formats

DSPy supports the following training data formats:

- `TrainDataFormat.CHAT`: For chat-based models (recommended)
- `TrainDataFormat.COMPLETION`: For completion-based models

## Monitoring and Cancellation

You can monitor the status of your fine-tuning job:

```python
status = job.status()
print(f"Current status: {status}")
```

And cancel it if needed:

```python
job.cancel()
```

## Next Steps

For more advanced fine-tuning techniques, check out the DSPy tutorials on fine-tuning and the BetterTogether optimizer, which combines fine-tuning with prompt optimization.