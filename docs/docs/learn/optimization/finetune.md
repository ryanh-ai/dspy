# Fine-tuning Models in DSPy

DSPy provides a unified interface for fine-tuning language models across different providers. This guide explains how to use DSPy's fine-tuning capabilities with various model providers.

## Overview

Fine-tuning allows you to adapt pre-trained language models to your specific tasks and data. DSPy supports fine-tuning with the following providers:

- OpenAI
- AWS Bedrock (including Amazon Nova models)
- Databricks

## Basic Usage

To fine-tune a model in DSPy, you need:

1. A base model to fine-tune
2. Training data in the appropriate format
3. Provider-specific configuration

Here's a basic example:

```python
import dspy
from dspy.clients.utils_finetune import TrainDataFormat

# Enable experimental features (required for fine-tuning)
dspy.settings.experimental = True

# Create an LM instance with the model you want to fine-tune
lm = dspy.LM(model="bedrock/amazon.nova-lite-v1")

# Prepare your training data
train_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
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
        # Provider-specific parameters
        "s3_path": "s3://your-bucket/fine-tuning-data/",
        "aws_profile": "default",  # Optional
        "hyperparameters": {
            "epochCount": 3,
            "batchSize": 4,
            "learningRate": 1e-5
        }
    }
)

# Wait for the job to complete
finetuned_model = job.result()

# Use the fine-tuned model
finetuned_lm = dspy.LM(model=finetuned_model)
response = finetuned_lm("What is the capital of Germany?")
print(response)
```

## Data Formats

DSPy supports two main data formats for fine-tuning:

### Chat Format

For chat models, use the `TrainDataFormat.CHAT` format:

```python
chat_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great, thank you!"}
        ]
    }
]
```

### Completion Format

For completion models, use the `TrainDataFormat.COMPLETION` format:

```python
completion_data = [
    {
        "prompt": "What is the capital of France?",
        "completion": "Paris is the capital of France."
    }
]
```

## Provider-Specific Configuration

### OpenAI

For OpenAI models, you can specify hyperparameters like:

```python
train_kwargs = {
    "n_epochs": 3,
    "batch_size": 4,
    "learning_rate_multiplier": 1.0
}
```

### AWS Bedrock

For AWS Bedrock models (including Nova Lite and Nova Micro models), you need to provide:

```python
train_kwargs = {
    "s3_path": "s3://your-bucket/fine-tuning-data/",  # Required
    "aws_profile": "default",  # Optional, uses default credentials if not provided
    "hyperparameters": {
        "epochCount": 3,
        "batchSize": 4,
        "learningRate": 1e-5
    },
    "job_name": "my-finetune-job",  # Optional
    "start_endpoint": True  # Optional, whether to start an endpoint after training
}
```

When using Bedrock, you can include a system prompt in your training data:

```python
chat_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in geography."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
]
```

### Databricks

For Databricks models, you need to provide:

```python
train_kwargs = {
    "train_data_path": "/Volumes/catalog/schema/volume/path",  # Required
    "register_to": "catalog.schema.model_name",  # Required
    "task_type": "CHAT_COMPLETION",  # or "INSTRUCTION_FINETUNE"
    "databricks_host": "https://your-workspace.cloud.databricks.com",  # Optional
    "databricks_token": "your-token",  # Optional
    "skip_deploy": False  # Optional
}
```

## Managing Fine-tuning Jobs

### Checking Job Status

You can check the status of a fine-tuning job:

```python
status = job.status()
print(f"Job status: {status}")
```

### Cancelling a Job

To cancel a running fine-tuning job:

```python
job.cancel()
```

### Managing Endpoints (Bedrock)

For AWS Bedrock, you can start and stop endpoints for your fine-tuned models:

```python
# Start an endpoint
job.start_endpoint()

# Stop an endpoint
job.stop_endpoint()
```

## Default Values

If not specified, DSPy will use reasonable defaults for hyperparameters:

- For Bedrock:
  - `epochCount`: 3
  - `batchSize`: 4
  - `learningRate`: 1e-5

## Advanced Usage with BootstrapFinetune

For more advanced fine-tuning, DSPy provides the `BootstrapFinetune` optimizer, which can fine-tune models within a DSPy program:

```python
import dspy
from dspy.teleprompt import BootstrapFinetune

# Define your program
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)

# Create instances
base_lm = dspy.LM("bedrock/amazon.nova-lite-v1")
program = SimpleQA()

# Attach the LM to the program
for predictor in program.predictors():
    predictor.lm = base_lm

# Create the optimizer
optimizer = BootstrapFinetune(
    train_kwargs={
        "s3_path": "s3://your-bucket/fine-tuning-data/",
        "aws_profile": "default",
        "hyperparameters": {
            "epochCount": 3,
            "batchSize": 4,
            "learningRate": 1e-5
        }
    }
)

# Compile the program
optimized_program = optimizer.compile(
    student=program,
    trainset=train_examples
)

# Use the optimized program
result = optimized_program(question="What is the capital of France?")
print(result.answer)
```

## Conclusion

Fine-tuning models in DSPy provides a powerful way to adapt language models to your specific tasks. By leveraging DSPy's unified interface, you can easily fine-tune models across different providers without changing your code structure.

For more examples and advanced usage, check out the tutorials in the DSPy documentation.