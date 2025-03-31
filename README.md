# llm-bedrock-anthropic

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock-anthropic.svg)](https://pypi.org/project/llm-bedrock-anthropic/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sblakey/llm-bedrock-anthropic/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Anthropic's Claude models.

## New: claude opus model 
### claude-3-opus available on us-west-2

## Installation

Install this plugin in the same environment as LLM. From the current directory
```bash
llm install llm-bedrock-anthropic
```
## Configuration

You will need to specify AWS Configuration with the normal boto3 and environment variables.

For example, to use the region `us-west-2` and AWS credentials under the `personal` profile, set the environment variables

```bash
export AWS_DEFAULT_REGION=us-west-2
export AWS_PROFILE=personal
```

## Usage

This plugin adds models called `bedrock-claude` and `bedrock-claude-instant`.

You can query them like this:

```bash
llm -m bedrock-claude-instant "Ten great names for a new space station"
```

```bash
llm -m bedrock-claude "Compare and contrast the leadership styles of Abraham Lincoln and Boris Johnson."
```

## Options

- `max_tokens_to_sample`, default 8_191: The maximum number of tokens to generate before stopping

Use like this:
```bash
llm -m bedrock-claude -o max_tokens_to_sample 20 "Sing me the alphabet"
 Here is the alphabet song:

A B C D E F G
H I J
```

- `bedrock_model_id`: allows you to override the `model_id` passed to Bedrock.

Can be a base model ID or ARN from https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns

Can also be an ARN of provisioned throughput for a base or custom model.

Use it like this:
```bash
llm -m bedrock-claude-haiku -o bedrock_model_id anthropic.claude-3-sonnet-20240229-v1:0 "Remind me how to post a new version to PyPI"
```
- `bedrock_attach`: allows you to attach one or more (comma-separated) image or document files to your prompt.

See the Amazon Bedrock documentation for details about which file types are supported with which models: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html

Use it like this:
```bash
llm -m bedrock-claude-haiku -o bedrock_attach cat.jpg "Describe this picture."
```
```bash
llm -m bedrock-claude-haiku -o bedrock_attach cat.jpg,mouse.jpg "What do these animals have in common?"
```
```bash
llm -m bedrock-claude-haiku -o bedrock_attach Model_Card_Claude_3_Addendum.pdf "Summarize this document."
```

The newer command line parameters `-a`, `--attachment`, `--at`, and `--attachment-type` are also supported.
The `bedrock_atach` option is retained for backwards compatibility, and will be removed in the next major version.
