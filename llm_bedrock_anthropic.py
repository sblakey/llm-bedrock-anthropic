from typing import Optional, List

import json
import mimetypes

import boto3
import llm
from pydantic import Field, field_validator


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
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-5-sonnet-20240620-v1:0"),
        aliases=(
            "bedrock-claude-v3.5-sonnet",
            "bedrock-claude-sonnet",
            "bedrock-sonnet",
            "bedrock-claude",
            "bc",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-opus-20240229-v1:0"),
        aliases=(
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
        # TODO: Make the defaults model-specific.
        max_tokens_to_sample: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4096,  # Bedrock complained when I passed a higher number into claudev3.5 Sonnet.
        )
        bedrock_model_id: Optional[str] = Field(
            description="Bedrock modelId or ARN of base, custom, or provisioned model",
            default=None,
        )
        bedrock_attach_file: Optional[str] = Field(
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
    def prompt_to_content(prompt):
        """
        Convert a llm.Prompt object to the content format expected by the Bedrock Converse API.
        If we encounter the bedrock_attach_file option, detect the file type and use the
        proper Bedrock Converse content type to attach the file to the prompt.

        :param prompt: A llm Prompt objet.
        :return: A content object that conforms to the Bedrock Converse API.
        """
        content = []
        if prompt.options.bedrock_attach_file:
            mime_type, _ = mimetypes.guess_type(prompt.options.bedrock_image_file)
            if not mime_type:
                raise ValueError(
                    f"Unable to guess mime type for file: {prompt.options.bedrock_attach_file}"
                )

            file_type, file_format = mime_type.split("/")
            if file_type != "image":
                raise ValueError(
                    f"Unsupported file type for file: {prompt.options.bedrock_attach_file}"
                )
            if file_format not in ['png', 'jpeg', 'gif', 'webp']:
                raise ValueError(
                    f"Unsupported image format for file: {prompt.options.bedrock_attach_file}"
                )

            with open(prompt.options.bedrock_image_file, "rb") as fp:
                source_bytes = fp.read()

            content.append(
                {
                    'image': {
                        'format': file_format,
                        'source': {
                            'bytes': source_bytes
                        }
                    }
                }
            )

        # Append the prompt text as a text content block.
        content.append(
            {
                'text': prompt.prompt
            }
        )

        return content

    @staticmethod
    def build_messages(prompt_content, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                if (
                    'response_json' in response and
                    response.response_json and
                    'bedrock_user_content' in response.response_json
                ):
                    user_content = response.response_json['bedrock_user_content']
                else:
                    user_content = [
                        {
                            'text': response.prompt.prompt
                        }
                    ]
                assistant_content = [
                    {
                        'text': response.text()
                    }
                ]
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": user_content
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        },
                    ]
                )

        messages.append({"role": "user", "content": prompt_content})
        return messages

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
        bedrock_model_id = prompt.options.bedrock_model_id or self.model_id

        if prompt.system and self.model_id in [
            "anthropic.claude-v2",
            "anthropic.claude-instant-v1",
        ]:
            prompt.prompt = prompt.system + "\n" + prompt.prompt

        prompt_content = self.prompt_to_content(prompt)
        messages = self.build_messages(prompt_content, conversation)

        # Preserve the Bedrock-specific user content dict so it can be re-used in
        # future conversations.
        response.response_json = {
            'bedrock_user_content': prompt_content
        }

        inference_config = {
            'maxTokens': prompt.options.max_tokens_to_sample
        }

        # Put together parameters for the Bedrock Converse API.
        params = {
            'modelId': bedrock_model_id,
            'messages': messages,
            'inferenceConfig': inference_config,
        }

        if prompt.system:
            params['system'] = [
                {
                    'text': prompt.system
                }
            ]

        client = boto3.client('bedrock-runtime')
        if stream:
            bedrock_response = client.converse_stream(**params)
            for event in bedrock_response['stream']:
                (event_type, event_content), = event.items()
                if event_type == "contentBlockDelta":
                    completion = event_content["delta"]["text"]
                    yield completion
        else:
            bedrock_response = client.converse(**params)
            completion = bedrock_response['output']['message']['content'][-1]['text']
            yield completion
