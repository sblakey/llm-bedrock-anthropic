# llm-bedrock-claude

Plugin for [LLM](https://llm.datasette.io/) adding support for Anthropic's Claude models.

## Installation

Install this plugin in the same environment as LLM. From the current directory
```bash
llm install -e .
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
