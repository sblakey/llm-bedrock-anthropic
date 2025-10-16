# Imports

import mimetypes
import os
from base64 import b64decode, b64encode
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import boto3
import llm
from PIL import Image
from pydantic import Field, field_validator


@dataclass
class AttachmentData:
    mime_type: str
    content: Union[Path, bytes]
    name: Optional[str] = None

    @property
    def is_file_path(self) -> bool:
        return isinstance(self.content, Path)

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    @property
    def is_document(self) -> bool:
        return self.mime_type in MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT


# Constants

# See: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
BEDROCK_CONVERSE_IMAGE_FORMATS = ["png", "jpeg", "gif", "webp"]
MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

# See: https://docs.anthropic.com/en/docs/build-with-claude/vision
ANTHROPIC_MAX_IMAGE_LONG_SIZE = 1568


# Much of this code is derived from https://github.com/tomviner/llm-claude


@llm.hookimpl
def register_models(register):
    # Claude 2 and earlier models (no attachment support)
    register(
        BedrockClaude("anthropic.claude-instant-v1", supports_attachments=False),
        aliases=("bedrock-claude-instant", "bci"),
    )
    register(
        BedrockClaude("anthropic.claude-v2", supports_attachments=False),
        aliases=("bedrock-claude-v2-0",),
    )
    register(
        BedrockClaude("anthropic.claude-v2:1", supports_attachments=False),
        aliases=(
            "bedrock-claude-v2.1",
            "bedrock-claude-v2",
        ),
    )

    # Claude 3 models (with attachment support)
    register(
        BedrockClaude(
            "anthropic.claude-3-sonnet-20240229-v1:0", supports_attachments=True
        ),
        aliases=("bedrock-claude-v3-sonnet",),
    )
    register(
        BedrockClaude(
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v3.5-sonnet-v2",
            "bedrock-claude-sonnet-v2",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-3-5-sonnet-20240620-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v3.5-sonnet",
            "bedrock-claude-sonnet",
            "bedrock-sonnet",
            "bedrock-claude",
            "bc",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-3-opus-20240229-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v3-opus",
            "bedrock-claude-opus",
            "bedrock-opus",
            "bo",
        ),
    )
    register(
        BedrockClaude(
            "us.anthropic.claude-3-5-haiku-20241022-v1:0", supports_attachments=False
        ),
        aliases=(
            "bedrock-claude-v3.5-haiku",
            "bedrock-haiku-v3.5",
            "bh-v3.5",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-3-haiku-20240307-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v3-haiku",
            "bedrock-claude-haiku",
            "bedrock-haiku",
            "bh",
        ),
    )
    register(
        BedrockClaude(
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v3.7-sonnet",
            "bedrock-claude-sonnet-v3.7",
            "bc-v3.7",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-sonnet-4-20250514-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v4-sonnet",
            "bedrock-claude-sonnet-v4",
            "bc-v4",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-opus-4-20250514-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v4-opus",
            "bedrock-claude-opus-v4",
            "bo-v4",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-opus-4-1-20250805-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v4.1-opus",
            "bedrock-claude-opus-v4.1",
            "bo-v4.1",
        ),
    )
    register(
        BedrockClaude(
            "anthropic.claude-sonnet-4-5-20250929-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v4.5-sonnet",
            "bedrock-claude-sonnet-v4.5",
            "bc-v4.5",
        ),
    )
    register(
        BedrockClaude(
            "us.anthropic.claude-opus-4-1-20250805-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v4.1-opus-us",
            "bedrock-claude-opus-v4.1-us",
            "bo-v4.1-us",
        ),
    )
    register(
        BedrockClaude(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0", supports_attachments=True
        ),
        aliases=(
            "bedrock-claude-v4.5-sonnet-us",
            "bedrock-claude-sonnet-v4.5-us",
            "bc-v4.5-us",
        ),
    )
    register(
        BedrockClaude(
            "global.anthropic.claude-sonnet-4-20250514-v1:0",
            supports_attachments=True,
        ),
        aliases=(
            "bedrock-claude-v4-sonnet-global",
            "bedrock-claude-sonnet-v4-global",
            "bc-v4-global",
        ),
    )
    register(
        BedrockClaude(
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            supports_attachments=True,
        ),
        aliases=(
            "bedrock-claude-v4.5-sonnet-global",
            "bedrock-claude-sonnet-v4.5-global",
            "bc-v4.5-global",
        ),
    )


class BedrockClaude(llm.Model):
    can_stream: bool = True

    # TODO: expose other Options
    class Options(llm.Options):
        # TODO: Make the defaults model-specific.
        max_tokens_to_sample: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4096,  # Bedrock complained when I passed a higher number into claude v3.5 Sonnet.
        )
        bedrock_model_id: Optional[str] = Field(
            description="Bedrock modelId or ARN of base, custom, or provisioned model",
            default=None,
        )
        bedrock_attach: Optional[str] = Field(
            description="Attach the given image or document file (or files, separated by comma) to the prompt.",
            default=None,
        )

        @field_validator("max_tokens_to_sample")
        def validate_length(cls, max_tokens_to_sample):
            if not (0 < max_tokens_to_sample <= 1_000_000):
                raise ValueError("max_tokens_to_sample must be in range 1-1,000,000")
            return max_tokens_to_sample

    def __init__(self, model_id, supports_attachments=False):
        self.model_id = model_id
        self.supports_attachments = supports_attachments
        if supports_attachments:
            image_mime_types = {
                f"image/{fmt}" for fmt in BEDROCK_CONVERSE_IMAGE_FORMATS
            }
            document_mime_types = set(
                MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT.keys()
            )
            self.attachment_types = image_mime_types.union(document_mime_types)

    @staticmethod
    def load_and_preprocess_image(file):
        """
        Load and pre-process the given image for use with Anthropic models and the Bedrock
        Converse API:
        * Resize if needed.
        * Convert into a supported format if needed.
        * Do nothing if the image is already compatible.
        Even if Bedrock can resize images for us, we do this here to avoid unnecessary
        bandwidth and to support additional image file types.

        :param file: An image file path.
        :return: A bytes, image_format tuple containing the resulting image data and format.
                 Use the original data/format if possible, and choose an appropriate format if
                 the image needed to be resized.
        """
        with open(file, "rb") as fp:
            img_bytes = fp.read()

        with Image.open(BytesIO(img_bytes)) as img:
            img_format = img.format
            width, height = img.size
            if (
                width > ANTHROPIC_MAX_IMAGE_LONG_SIZE
                or height > ANTHROPIC_MAX_IMAGE_LONG_SIZE
            ):
                # Resize the image while preserving the aspect ratio
                img.thumbnail(
                    (ANTHROPIC_MAX_IMAGE_LONG_SIZE, ANTHROPIC_MAX_IMAGE_LONG_SIZE)
                )

            # Change format if necessary
            if (
                img_format.lower() in BEDROCK_CONVERSE_IMAGE_FORMATS
                and img.size == (width, height)  # Original size, no resize needed
            ):
                return img_bytes, img_format.lower()

            # Re-export the image with the appropriate format
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue(), "png"

    def image_path_to_content_block(self, path):
        """
        Create a Bedrock Converse content block out of the given image file path.
        :param path: A file path to an image file.
        :return: A Bedrock Converse API content block containing the image.
        """
        source_bytes, file_format = self.load_and_preprocess_image(path)

        return {"image": {"format": file_format, "source": {"bytes": source_bytes}}}

    @staticmethod
    def sanitize_file_name(file_path):
        """
        Generate a file name out of the given file path that conforms to the Bedrock
        Converse API conventions:
        * Alphanumeric characters
        * Whitespace characters (no more than one in a row)
        * Hyphens
        * Parentheses
        * Square brackets
        * Maximum length of 200.
        See also: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
        :param file_path:
        :return:
        """
        head, tail = os.path.split(file_path)
        name, ext = os.path.splitext(tail)

        # Sanitize the name part
        sanitized_name = ""
        for c in name:
            if (
                c
                in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_()[]"
            ):
                sanitized_name += c
            else:
                sanitized_name += "_"

        if not sanitized_name:
            sanitized_name = "file"

        # Ensure total length (including extension) is under 200
        max_name_length = 200 - len(ext)
        sanitized_name = sanitized_name[:max_name_length]

        return sanitized_name + ext

    def document_path_to_content_block(self, file_path, mime_type):
        """
        Create a Bedrock Converse content block out of the given document file path.
        :param file_path: A file path to a document file.
        :param mime_type: The file’s MIME type.
        :return: A Bedrock Converse API content block containing the document.
        """
        with open(file_path, "rb") as fp:
            source_bytes = fp.read()
        return {
            "document": {
                "format": MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                "name": self.sanitize_file_name(file_path),
                "source": {"bytes": source_bytes},
            }
        }

    def validate_attachment(self, data: AttachmentData) -> None:
        """Validate attachments"""
        if not (data.is_image or data.is_document):
            raise ValueError(f"Unsupported attachment type: {data.mime_type}")

    def process_attachment(self, data: AttachmentData) -> dict:
        """
        Process attachments and generate content blocks for Bedrock Converse API
        """
        self.validate_attachment(data)

        if data.is_image:
            if data.is_file_path:
                return self.image_path_to_content_block(str(data.content))
            return self.image_bytes_to_content_block(data.content, data.mime_type)

        if data.is_document:
            if data.is_file_path:
                return self.document_path_to_content_block(
                    str(data.content), data.mime_type
                )
            return self.document_bytes_to_content_block(
                data.content, data.mime_type, data.name
            )

    def create_attachment_data(self, attachment) -> AttachmentData:
        """Generate AttachmentData from a modern attachment"""
        if hasattr(attachment, "path") and attachment.path is not None:
            return AttachmentData(
                mime_type=attachment.type, content=Path(attachment.path)
            )

        return AttachmentData(
            mime_type=attachment.type,
            content=attachment.content_bytes(),
            name="attachment",
        )

    def image_bytes_to_content_block(self, image_bytes: bytes, mime_type: str) -> dict:
        """
        Create a Bedrock Converse content block from image bytes.
        :param image_bytes: The raw image bytes
        :param mime_type: The MIME type of the image
        :return: A Bedrock Converse API content block containing the image
        """
        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size
            if (
                width > ANTHROPIC_MAX_IMAGE_LONG_SIZE
                or height > ANTHROPIC_MAX_IMAGE_LONG_SIZE
            ):
                # Resize the image while preserving the aspect ratio
                img.thumbnail(
                    (ANTHROPIC_MAX_IMAGE_LONG_SIZE, ANTHROPIC_MAX_IMAGE_LONG_SIZE)
                )

                # Re-export the image
                with BytesIO() as buffer:
                    img.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                    file_format = "png"
            else:
                # Use original format from mime_type
                file_format = mime_type.split("/")[-1]
                if file_format not in BEDROCK_CONVERSE_IMAGE_FORMATS:
                    # Convert to PNG if format not supported
                    with BytesIO() as buffer:
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        file_format = "png"

        return {"image": {"format": file_format, "source": {"bytes": image_bytes}}}

    def document_bytes_to_content_block(
        self, doc_bytes: bytes, mime_type: str, name: Optional[str] = None
    ) -> dict:
        """
        Create a Bedrock Converse content block from document bytes.
        :param doc_bytes: The raw document bytes
        :param mime_type: The MIME type of the document
        :param name: Optional name for the document
        :return: A Bedrock Converse API content block containing the document
        """
        if name is None:
            name = "document"

        return {
            "document": {
                "format": MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                "name": self.sanitize_file_name(name),
                "source": {"bytes": doc_bytes},
            }
        }

    def prompt_to_content(self, prompt):
        """
        Convert a llm.Prompt object to the content format expected by the Bedrock Converse API.
        If we encounter the bedrock_attach_files option, detect the file type(s) and use the
        proper Bedrock Converse content type to attach the file(s) to the prompt.

        :param prompt: A llm Prompt objet.
        :return: A content object that conforms to the Bedrock Converse API.
        """
        content = []

        # Legacy attachments with -o bedrock_attach
        if prompt.options.bedrock_attach:
            # Support multiple files separated by comma.
            for file_path in prompt.options.bedrock_attach.split(","):
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    raise ValueError(f"Unable to guess mime type for file: {file_path}")

                file_path = os.path.expanduser(file_path)
                if mime_type.startswith("image/"):
                    content.append(self.image_path_to_content_block(file_path))
                elif mime_type in MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT:
                    content.append(
                        self.document_path_to_content_block(file_path, mime_type)
                    )
                else:
                    raise ValueError(f"Unsupported file type for file: {file_path}")

        # Modern attachments with -a or --attachment
        if hasattr(prompt, "attachments"):
            data = [self.create_attachment_data(a) for a in prompt.attachments]
            content_blocks = [self.process_attachment(d) for d in data]
            content.extend(content_blocks)

        # Append the prompt text as a text content block.
        content.append({"text": prompt.prompt})

        return content

    def encode_bytes(self, o):
        """
        Recursively replace any "bytes" dict attribute in the given object with a base64
        encoded value as "bytes_b64". This is done to preserve the data during logging activities.

        :param o: A Python object.
        :return: A copy of the input, but with all "bytes" keys in dicts replaces by base64
                 encoded values names "bytes".
        """
        if isinstance(o, list):
            return [self.encode_bytes(i) for i in o]
        elif isinstance(o, dict):
            result = {}
            for key, value in o.items():
                if key == "bytes":
                    result["bytes_b64"] = b64encode(value).decode("utf-8")
                else:
                    result[key] = self.encode_bytes(value)
            return result
        else:
            return o

    def decode_bytes(self, o):
        """
        Recursively replace any "bytes_b64" dict attribute in the given object with a
        base64 decoded value as "bytes". This is the reverse of the above, so the resulting
        data can be sent to Bedrock in its expected form.

        :param o: A Python object.
        :return: A copy of the input, but with all "bytes_b64" keys in dicts replaced by base64
                 decoded values names "bytes".
        """
        if isinstance(o, list):
            return [self.decode_bytes(i) for i in o]
        elif isinstance(o, dict):
            result = {}
            for key, value in o.items():
                if key == "bytes_b64":
                    result["bytes"] = b64decode(value)
                else:
                    result[key] = self.decode_bytes(value)
            return result
        else:
            return o

    def build_messages(self, prompt_content, conversation) -> List[dict]:
        messages = []
        if conversation:
            for response in conversation.responses:
                if (
                    response.response_json
                    and "bedrock_user_content" in response.response_json
                ):
                    user_content = self.decode_bytes(
                        response.response_json["bedrock_user_content"]
                    )
                else:
                    user_content = [{"text": response.prompt.prompt}]
                assistant_content = [{"text": response.text()}]
                messages.extend(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                )

        messages.append({"role": "user", "content": prompt_content})
        return messages

    def execute(self, prompt, stream, response, conversation):
        # Claude 2.0 and Claude Instant did not historically really support system prompts:
        # https://docs.anthropic.com/claude/docs/constructing-a-prompt#system-prompt-optional
        #
        # As of the release of the Messages API, this seems like it has been fixed
        # https://docs.anthropic.com/claude/docs/system-prompts, but it is not documented that
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

        # Preserve the Bedrock-specific user content dict, so it can be re-used in
        # future conversations.
        response.response_json = {
            "bedrock_user_content": self.encode_bytes(prompt_content)
        }

        inference_config = {"maxTokens": prompt.options.max_tokens_to_sample}

        # Put together parameters for the Bedrock Converse API.
        params = {
            "modelId": bedrock_model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if prompt.system:
            params["system"] = [{"text": prompt.system}]

        client = boto3.client("bedrock-runtime")
        if stream:
            bedrock_response = client.converse_stream(**params)
            response.response_json |= bedrock_response
            events = []
            for event in bedrock_response["stream"]:
                ((event_type, event_content),) = event.items()
                if event_type == "contentBlockDelta":
                    completion = event_content["delta"]["text"]
                    yield completion
                events.append(event)
            response.response_json["stream"] = events
        else:
            bedrock_response = client.converse(**params)
            completion = bedrock_response["output"]["message"]["content"][-1]["text"]
            response.response_json |= bedrock_response
            yield completion
        self.set_usage(response)

    def set_usage(self, response: llm.Response):
        if not hasattr(response, "set_usage"):
            # Older versions of llm do not have this method
            return
        res_json = response.response_json

        if "usage" in res_json:
            response.set_usage(
                input=res_json["usage"]["inputTokens"],
                output=res_json["usage"]["outputTokens"],
            )
        elif "stream" in res_json:
            events = res_json["stream"]
            for event in events:
                ((event_type, event_content),) = event.items()
                if event_type == "metadata":
                    input_tokens = event_content["usage"]["inputTokens"]
                    output_tokens = event_content["usage"]["outputTokens"]
                    response.set_usage(input=input_tokens, output=output_tokens)
                    break
