# Imports

from typing import Optional, List, Union
from dataclasses import dataclass

import mimetypes
from base64 import b64encode, b64decode
from io import BytesIO
import os
from pathlib import Path

import boto3
import llm
from pydantic import Field, field_validator
from PIL import Image

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
        aliases=("bedrock-claude-v2-0",)
    )
    register(
        BedrockClaude("anthropic.claude-v2:1", supports_attachments=False),
        aliases=("bedrock-claude-v2.1", "bedrock-claude-v2",),
    )
    
    # Claude 3 models (with attachment support)
    register(
        BedrockClaude("anthropic.claude-3-sonnet-20240229-v1:0", supports_attachments=True),
        aliases=(
            "bedrock-claude-v3-sonnet",
        ),
    )
    register(
        BedrockClaude("us.anthropic.claude-3-5-sonnet-20241022-v2:0", supports_attachments=True),
        aliases=(
            "bedrock-claude-v3.5-sonnet-v2",
            "bedrock-claude-sonnet-v2",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-5-sonnet-20240620-v1:0", supports_attachments=True),
        aliases=(
            "bedrock-claude-v3.5-sonnet",
            "bedrock-claude-sonnet",
            "bedrock-sonnet",
            "bedrock-claude",
            "bc",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-opus-20240229-v1:0", supports_attachments=True),
        aliases=(
            "bedrock-claude-v3-opus",
            "bedrock-claude-opus",
            "bedrock-opus",
            "bo",
        ),
    )
    register(
        BedrockClaude("us.anthropic.claude-3-5-haiku-20241022-v1:0", supports_attachments=False),
        aliases=(
            "bedrock-claude-v3.5-haiku",
            "bedrock-haiku-v3.5",
            "bh-v3.5",
        ),
    )
    register(
        BedrockClaude("anthropic.claude-3-haiku-20240307-v1:0", supports_attachments=True),
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
            image_mime_types = {f"image/{fmt}" for fmt in BEDROCK_CONVERSE_IMAGE_FORMATS}
            document_mime_types = set(MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT.keys())
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
            if width > ANTHROPIC_MAX_IMAGE_LONG_SIZE or height > ANTHROPIC_MAX_IMAGE_LONG_SIZE:
                # Resize the image while preserving the aspect ratio
                img.thumbnail((ANTHROPIC_MAX_IMAGE_LONG_SIZE, ANTHROPIC_MAX_IMAGE_LONG_SIZE))

            # Change format if necessary
            if (
                img_format.lower() in BEDROCK_CONVERSE_IMAGE_FORMATS and
                img.size == (width, height)  # Original size, no resize needed
            ):
                return img_bytes, img_format.lower()

            # Re-export the image with the appropriate format
            with BytesIO() as buffer:
                img.save(buffer, format='PNG')
                return buffer.getvalue(), 'png'

    def image_path_to_content_block(self, path):
        """
        Create a Bedrock Converse content block out of the given image file path.
        :param path: A file path to an image file.
        :return: A Bedrock Converse API content block containing the image.
        """
        source_bytes, file_format = self.load_and_preprocess_image(path)

        return {
            'image': {
                'format': file_format,
                'source': {
                    'bytes': source_bytes
                }
            }
        }

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
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_()[]":
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
            'document': {
                'format': MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                'name': self.sanitize_file_name(file_path),
                'source': {
                    'bytes': source_bytes
                }
            }
        }

    def validate_attachment(self, data: AttachmentData) -> None:
        """添付ファイルのバリデーションを行う"""
        if not (data.is_image or data.is_document):
            raise ValueError(f"Unsupported attachment type: {data.mime_type}")

    def process_attachment(self, data: AttachmentData) -> dict:
        """
        添付ファイルを処理してBedrock Converse API用のcontent blockを生成
        """
        self.validate_attachment(data)
        
        if data.is_image:
            if data.is_file_path:
                return self.image_path_to_content_block(str(data.content))
            return self.image_bytes_to_content_block(data.content, data.mime_type)
            
        if data.is_document:
            if data.is_file_path:
                return self.document_path_to_content_block(str(data.content), data.mime_type)
            return self.document_bytes_to_content_block(data.content, data.mime_type, data.name)

    def create_attachment_data(self, attachment) -> AttachmentData:
        """Generate AttachmentData from a modern attachment"""
        if hasattr(attachment, "path") and attachment.path is not None:
            return AttachmentData(
                mime_type=attachment.type,
                content=Path(attachment.path)
            )
        
        return AttachmentData(
            mime_type=attachment.type,
            content=attachment.content_bytes(),
            name="attachment"
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
            if width > ANTHROPIC_MAX_IMAGE_LONG_SIZE or height > ANTHROPIC_MAX_IMAGE_LONG_SIZE:
                # Resize the image while preserving the aspect ratio
                img.thumbnail((ANTHROPIC_MAX_IMAGE_LONG_SIZE, ANTHROPIC_MAX_IMAGE_LONG_SIZE))
                
                # Re-export the image
                with BytesIO() as buffer:
                    img.save(buffer, format='PNG')
                    image_bytes = buffer.getvalue()
                    file_format = 'png'
            else:
                # Use original format from mime_type
                file_format = mime_type.split('/')[-1]
                if file_format not in BEDROCK_CONVERSE_IMAGE_FORMATS:
                    # Convert to PNG if format not supported
                    with BytesIO() as buffer:
                        img.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                        file_format = 'png'

        return {
            'image': {
                'format': file_format,
                'source': {
                    'bytes': image_bytes
                }
            }
        }

    def document_bytes_to_content_block(self, doc_bytes: bytes, mime_type: str, name: Optional[str] = None) -> dict:
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
            'document': {
                'format': MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                'name': self.sanitize_file_name(name),
                'source': {
                    'bytes': doc_bytes
                }
            }
        }
