import os
import pytest
import mimetypes
from PIL import Image
from io import BytesIO
from pathlib import Path

from llm_bedrock_anthropic import (
    BedrockClaude, 
    ANTHROPIC_MAX_IMAGE_LONG_SIZE, 
    AttachmentData, 
    MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT
)

# Test fixtures paths
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
IMAGES_DIR = os.path.join(FIXTURES_DIR, 'images')
DOCS_DIR = os.path.join(FIXTURES_DIR, 'documents')

class TestBedrockClaude(BedrockClaude):
    def execute(self, prompt, stream=False, response=None, **kwargs):
        # Mock implementation for testing
        return "Test response"
        
    def create_attachment_data_from_path(self, path):
        """Create AttachmentData from a file path"""
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for {path}")
        return AttachmentData(
            mime_type=mime_type,
            content=Path(path)
        )

@pytest.fixture
def model():
    return TestBedrockClaude("anthropic.claude-3-sonnet-20240229-v1:0", supports_attachments=True)

@pytest.fixture
def model_no_attachments():
    return TestBedrockClaude("anthropic.claude-v2", supports_attachments=False)

def test_attachment_data_properties():
    """Test properties of AttachmentData"""
    # Test with file path
    image_path = Path(os.path.join(IMAGES_DIR, 'test.png'))
    image_data = AttachmentData(
        mime_type='image/png', 
        content=image_path
    )
    
    assert image_data.is_file_path is True
    assert image_data.is_image is True
    assert image_data.is_document is False
    
    # Test with bytes
    doc_bytes = b'test document content'
    doc_data = AttachmentData(
        mime_type='text/plain', 
        content=doc_bytes,
        name='test.txt'
    )
    
    assert doc_data.is_file_path is False
    assert doc_data.is_image is False
    assert doc_data.is_document is True
    assert doc_data.name == 'test.txt'

def test_create_attachment_data(model):
    """Test creating AttachmentData from different sources"""
    class MockAttachment:
        def __init__(self, path=None, content=None, mime_type='image/png'):
            self.path = path
            self._content = content
            self.type = mime_type
        
        def content_bytes(self):
            return self._content

    # Test with file path
    image_path = os.path.join(IMAGES_DIR, 'test.png')
    file_attachment = MockAttachment(path=image_path)
    file_data = model.create_attachment_data(file_attachment)
    
    assert isinstance(file_data.content, Path)
    assert file_data.mime_type == 'image/png'
    assert file_data.name is None

    # Test with bytes
    byte_content = b'test image content'
    byte_attachment = MockAttachment(content=byte_content)
    byte_data = model.create_attachment_data(byte_attachment)
    
    assert byte_data.content == byte_content
    assert byte_data.mime_type == 'image/png'
    assert byte_data.name == 'attachment'

def test_create_attachment_data_from_path(model):
    """Test creating AttachmentData from file paths"""
    # Test image path
    image_path = os.path.join(IMAGES_DIR, 'test.png')
    image_data = model.create_attachment_data_from_path(image_path)
    
    assert isinstance(image_data.content, Path)
    assert image_data.mime_type == 'image/png'

    # Test document path
    doc_path = os.path.join(DOCS_DIR, 'test.txt')
    doc_data = model.create_attachment_data_from_path(doc_path)
    
    assert isinstance(doc_data.content, Path)
    assert doc_data.mime_type == 'text/plain'

def test_validate_attachment(model):
    """Test attachment validation"""
    # Valid image attachment
    image_data = AttachmentData(
        mime_type='image/png', 
        content=Path(os.path.join(IMAGES_DIR, 'test.png'))
    )
    model.validate_attachment(image_data)  # Should not raise exception

    # Valid document attachment
    doc_data = AttachmentData(
        mime_type='text/plain', 
        content=b'test document content'
    )
    model.validate_attachment(doc_data)  # Should not raise exception

    # Invalid attachment type
    with pytest.raises(ValueError, match="Unsupported attachment type"):
        invalid_data = AttachmentData(
            mime_type='application/x-unknown', 
            content=b'invalid content'
        )
        model.validate_attachment(invalid_data)

def test_process_attachment(model):
    """Test processing different types of attachments"""
    # Image file path
    image_path = os.path.join(IMAGES_DIR, 'test.png')
    image_data = AttachmentData(mime_type='image/png', content=Path(image_path))
    image_block = model.process_attachment(image_data)
    
    assert 'image' in image_block
    assert image_block['image']['format'] == 'png'
    assert 'bytes' in image_block['image']['source']

    # Image bytes - using a small valid PNG image
    image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82'
    image_bytes_data = AttachmentData(mime_type='image/png', content=image_bytes)
    image_bytes_block = model.process_attachment(image_bytes_data)
    
    assert 'image' in image_bytes_block
    assert image_bytes_block['image']['format'] == 'png'
    assert 'bytes' in image_bytes_block['image']['source']

    # Document file path
    doc_path = os.path.join(DOCS_DIR, 'test.txt')
    doc_data = AttachmentData(mime_type='text/plain', content=Path(doc_path))
    doc_block = model.process_attachment(doc_data)
    
    assert 'document' in doc_block
    assert doc_block['document']['format'] == 'txt'
    assert 'bytes' in doc_block['document']['source']

    # Document bytes
    doc_bytes = b'test document content'
    doc_bytes_data = AttachmentData(mime_type='text/plain', content=doc_bytes, name='test.txt')
    doc_bytes_block = model.process_attachment(doc_bytes_data)
    
    assert 'document' in doc_bytes_block
    assert doc_block['document']['format'] == 'txt'
    assert 'bytes' in doc_block['document']['source']
    assert doc_bytes_block['document']['name'] == 'test.txt'
