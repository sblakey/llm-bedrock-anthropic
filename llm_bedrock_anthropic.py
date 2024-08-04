from typing import Optional, List

import boto3
import llm
import json
from pydantic import Field, field_validator

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

# Much of this code is derived from https://github.com/tomviner/llm-claude


@llm.hookimpl
def register_models(register):
    register(
        BedrockClaude("anthropic.claude-instant-v1"),
        aliases=("bedrock-claude-instant", "bci"),
    )
    register(
        BedrockClaude("anthropic.claude-v2"), aliases=("bedrock-claude-v2-0",)
    )
    register(
        BedrockClaude("anthropic.claude-v2:1"),
        aliases=("bedrock-claude-v2.1", "bedrock-claude-v2",),
    )
    register(
        BedrockClaude("anthropic.claude-3-sonnet-20240229-v1:0"),
        aliases=(
            "bedrock-claude-v3-sonnet",
            "bedrock-claude-v3-sonnet",
            "bedrock-claude-sonnet",
            "bedrock-sonnet",
            "bedrock-claude",
            "bc",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-5-sonnet-20240620-v1:0"),
        aliases=(
            "bedrock-claude-v3.5-sonnet",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-opus-20240229-v1:0"),
        aliases=(
            "bedrock-claude-v3-opus",
            "bedrock-claude-v3-opus",
            "bedrock-claude-opus",
            "bedrock-opus",
            "bo",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-haiku-20240307-v1:0"),
        aliases=(
            "bedrock-claude-v3-haiku",
            "bedrock-claude-haiku",
            "bedrock-haiku",
            "bh",
        ),
    )    


class BedrockClaude(llm.Model):
    can_stream: bool = True

    # TODO: expose other Options
    class Options(llm.Options):
        max_tokens_to_sample: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=8191,  # Bedrock complained when I passed a higher number into claude-instant
        )
        bedrock_model_id: Optional[str] = Field(
            description="Bedrock modelId or ARN of base, custom, or provisioned model",
            default=None,
        )
        bedrock_image_file: Optional[str] = Field(
            description="Add the given image file to the prompt.",
            default=None,
        )

        @field_validator("max_tokens_to_sample")
        def validate_length(cls, max_tokens_to_sample):
            if not (0 < max_tokens_to_sample <= 1_000_000):
                raise ValueError("max_tokens_to_sample must be in range 1-1,000,000")
            return max_tokens_to_sample

    def __init__(self, model_id):
        self.model_id = model_id

    @staticmethod
    def build_messages(prompt, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": response.prompt.prompt,
                        },
                        {"role": "assistant", "content": response.text()},
                    ]
                )
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    @staticmethod
    def generate_prompt_messages_v3(self, prompt, conversation):
        def build_message(role, content):
            return {
                "role": role,
                "content": content
            }

        if conversation:
            for response in conversation.responses:
                yield build_message("user", response.prompt.prompt)
                yield build_message("assistant", response.text())

        yield build_message("user", prompt)

    def execute(self, prompt, stream, response, conversation):
        # Claude 2.0 and Claude Instant did not historically really support system prompts:
        # https://docs.anthropic.com/claude/docs/constructing-a-prompt#system-prompt-optional
        #
        # As of the release of the Messages API, this seems like it has been fixed
        # https://docs.anthropic.com/claude/docs/system-prompts but it is not documented that
        # Claude Instant and 2.0 support it (and the wording implies that it doesn't)
        # so what we do instead is put what would be the system prompt in the first line of the
        # `Human` prompt, as recommended in the documentation. This enables us to effectively use the
        #  `-s`, `-t` and `--save` flags.
        if prompt.system and self.model_id in [
            "anthropic.claude-v2",
            "anthropic.claude-instant-v1",
        ]:
            prompt.prompt = prompt.system + "\n" + prompt.prompt

        prompt.messages = self.build_messages(prompt, conversation)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": prompt.options.max_tokens_to_sample,
            "system": prompt.system or "",
            "messages": prompt.messages,
        }
        encoded_data = json.dumps(body)
        prompt.prompt_json = encoded_data

        client = boto3.client('bedrock-runtime')
        bedrock_model_id = prompt.options.bedrock_model_id or self.model_id
        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=bedrock_model_id, body=prompt.prompt_json
            )
            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                response = json.loads(chunk.get("bytes").decode())
                if response["type"] == "content_block_delta":
                    completion = response["delta"]["text"]
                    yield completion

        else:
            bedrock_response = client.invoke_model(
                modelId=bedrock_model_id, body=prompt.prompt_json
            )
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)
            completion = response.response_json["content"][-1]["text"]
            yield completion

    def execute_v3(self, prompt, stream, response, conversation):
        client = boto3.client('bedrock-runtime')

        messages = list(self.generate_prompt_messages_v3(prompt.prompt, conversation))
        prompt_json = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": prompt.options.max_tokens_to_sample,
            "messages": messages
        }
        if prompt.system:
            prompt_json.update({"system": prompt.system})
        prompt.prompt_json = prompt_json
        bedrock_model_id = prompt.options.bedrock_model_id or self.model_id
        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=bedrock_model_id, body=json.dumps(prompt_json)
            )
            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                if chunk:
                    response = json.loads(chunk.get("bytes").decode())
                    if response['type'] == 'content_block_delta':
                        if response["delta"]['type'] == 'text_delta':
                            yield response["delta"]["text"]
        else:
            bedrock_response = client.invoke_model(
                modelId=bedrock_model_id,
                body=json.dumps(prompt_json),
            )
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)
            completion = response.response_json["content"][0]["text"]
            yield completion
