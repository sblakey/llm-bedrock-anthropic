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
        aliases=("bedrock-claude-instant",),
    )
    register(BedrockClaude("anthropic.claude-v1"), aliases=("bedrock-claude-v1",))
    register(BedrockClaude("anthropic.claude-v2"), aliases=("bedrock-claude",))


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

    def execute(self, prompt, stream, response, conversation):
        client = boto3.client("bedrock-runtime")
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
