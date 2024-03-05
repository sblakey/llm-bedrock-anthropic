from typing import Optional

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
      aliases=("bedrock-claude-instant", 'bci'),
    )
    register(BedrockClaude("anthropic.claude-v1"), aliases=("bedrock-claude-v1",))
    register(BedrockClaude("anthropic.claude-v2"), aliases=('bedrock-claude-v2.0',))
    register(BedrockClaude("anthropic.claude-v2:1"),
             aliases=('bedrock-claude-v2.1', 'bedrock-claude-v2'))
    register(BedrockClaude("anthropic.claude-3-sonnet-20240229-v1:0"), aliases=('bedrock-claude-v3-sonnet', 'bedrock-claude', 'bc'))


class BedrockClaude(llm.Model):
    can_stream: bool = True

    # TODO: expose other Options
    class Options(llm.Options):
        max_tokens_to_sample: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=8191,  # Bedrock complained when I passed a higher number into claude-instant
        )

        @field_validator("max_tokens_to_sample")
        def validate_length(cls, max_tokens_to_sample):
            if not (0 < max_tokens_to_sample <= 1_000_000):
                raise ValueError("max_tokens_to_sample must be in range 1-1,000,000")
            return max_tokens_to_sample

    def __init__(self, model_id):
        self.model_id = model_id

    def generate_prompt_messages(self, prompt, conversation):
        if conversation:
            for response in conversation.responses:
                yield self.build_prompt(response.prompt.prompt, response.text())

        yield self.build_prompt(prompt)

    def build_prompt(self, human, ai=""):
        return f"{HUMAN_PROMPT}{human}{AI_PROMPT}{ai}"

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
        if self.model_id.startswith("anthropic.claude-3"):
            return self.execute_v3(prompt, stream, response, conversation)
        else:
            return self.execute_v1v2(prompt, stream, response, conversation)

    def execute_v1v2(self, prompt, stream, response, conversation):
        client = boto3.client('bedrock-runtime')

        # Claude 2 does not currently really support system prompts:
        # https://docs.anthropic.com/claude/docs/constructing-a-prompt#system-prompt-optional
        # so what we do instead is put what would be the system prompt in the first line of the
        # `Human` prompt, as recommended in the documentation. This enables us to effectively use the
        # `-s`, `-t` and `--save` flags. 
        if prompt.system:
            prompt.prompt = prompt.system + '\n' + prompt.prompt

        prompt_str = "".join(self.generate_prompt_messages(prompt.prompt, conversation))
        prompt_json = {
            "prompt": prompt_str,
            "max_tokens_to_sample": prompt.options.max_tokens_to_sample,
        }
        prompt.prompt_json = prompt_json
        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=self.model_id, body=json.dumps(prompt_json)
            )
            chunks = bedrock_response.get("body")

            for event in chunks:
                chunk = event.get("chunk")
                if chunk:
                    response = json.loads(chunk.get("bytes").decode())
                    completion = response["completion"]
                    yield completion

        else:
            bedrock_response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(prompt_json),
            )
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)
            completion = response.response_json["completion"]
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
        if stream:
            bedrock_response = client.invoke_model_with_response_stream(
                modelId=self.model_id, body=json.dumps(prompt_json)
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
                modelId=self.model_id,
                body=json.dumps(prompt_json),
            )
            body = bedrock_response["body"].read()
            response.response_json = json.loads(body)
            completion = response.response_json["content"][0]["text"]
            yield completion