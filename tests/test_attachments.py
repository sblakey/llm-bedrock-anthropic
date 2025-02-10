import os
import pytest
from PIL import Image
from io import BytesIO

from llm_bedrock_anthropic import BedrockClaude, ANTHROPIC_MAX_IMAGE_LONG_SIZE

# Test fixtures paths
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
IMAGES_DIR = os.path.join(FIXTURES_DIR, 'images')
DOCS_DIR = os.path.join(FIXTURES_DIR, 'documents')

@pytest.fixture
def model():
    return BedrockClaude("anthropic.claude-3-sonnet-20240229-v1:0", supports_attachments=True)

@pytest.fixture
def model_no_attachments():
    return BedrockClaude("anthropic.claude-v2", supports_attachments=False)

def test_attachment_support():
    """Test attachment support flag"""
    model_with = BedrockClaude("test-model", supports_attachments=True)
    model_without = BedrockClaude("test-model", supports_attachments=False)
    assert model_with.supports_attachments is True
    assert model_without.supports_attachments is False

def test_prompt_to_content_unsupported_model(model_no_attachments):
    """Test attempting to use attachments with unsupported model"""
    class MockPrompt:
        def __init__(self):
            self.prompt = "Describe this image"
            self.options = type('Options', (), {'bedrock_attach': 'test.png'})()
    
    with pytest.raises(ValueError, match="Model .* does not support attachments"):
        model_no_attachments.prompt_to_content(MockPrompt())

def test_load_and_preprocess_image_png(model):
    """Test loading and preprocessing a PNG image"""
    image_path = os.path.join(IMAGES_DIR, 'test.png')
    img_bytes, img_format = model.load_and_preprocess_image(image_path)
    
    # Verify the format
    assert img_format == 'png'
    
    # Verify the image can be loaded
    img = Image.open(BytesIO(img_bytes))
    assert img.format.lower() == 'png'
    assert img.size == (100, 100)

def test_load_and_preprocess_image_jpeg(model):
    """Test loading and preprocessing a JPEG image"""
    image_path = os.path.join(IMAGES_DIR, 'test.jpg')
    img_bytes, img_format = model.load_and_preprocess_image(image_path)
    
    # Verify the format
    assert img_format == 'jpeg'
    
    # Verify the image can be loaded
    img = Image.open(BytesIO(img_bytes))
    assert img.format.lower() == 'jpeg'
    assert img.size == (100, 100)

def test_load_and_preprocess_large_image(model):
    """Test loading and preprocessing an oversized image"""
    image_path = os.path.join(IMAGES_DIR, 'large_test.png')
    img_bytes, img_format = model.load_and_preprocess_image(image_path)
    
    # Verify the format
    assert img_format == 'png'
    
    # Verify the image was resized
    img = Image.open(BytesIO(img_bytes))
    width, height = img.size
    assert max(width, height) <= ANTHROPIC_MAX_IMAGE_LONG_SIZE

def test_image_path_to_content_block(model):
    """Test converting an image to a content block"""
    image_path = os.path.join(IMAGES_DIR, 'test.png')
    content_block = model.image_path_to_content_block(image_path)
    
    assert 'image' in content_block
    assert 'format' in content_block['image']
    assert content_block['image']['format'] == 'png'
    assert 'source' in content_block['image']
    assert 'bytes' in content_block['image']['source']

def test_sanitize_file_name(model):
    """Test file name sanitization"""
    test_cases = [
        ('test.pdf', 'test.pdf'),
        ('test file.pdf', 'test_file.pdf'),
        ('test@#$%^&*.pdf', 'test_______.pdf'),
        ('a' * 250 + '.pdf', ('a' * 196) + '.pdf'),  # 200 char limit
    ]
    
    for input_name, expected in test_cases:
        assert model.sanitize_file_name(input_name) == expected

def test_document_path_to_content_block(model):
    """Test converting a document to a content block"""
    doc_path = os.path.join(DOCS_DIR, 'test.txt')
    content_block = model.document_path_to_content_block(doc_path, 'text/plain')
    
    assert 'document' in content_block
    assert 'format' in content_block['document']
    assert content_block['document']['format'] == 'txt'
    assert 'name' in content_block['document']
    assert content_block['document']['name'] == 'test.txt'
    assert 'source' in content_block['document']
    assert 'bytes' in content_block['document']['source']

def test_prompt_to_content_single_image(model):
    """Test creating content with a single image attachment"""
    class MockPrompt:
        def __init__(self):
            self.prompt = "Describe this image"
            self.options = type('Options', (), {'bedrock_attach': os.path.join(IMAGES_DIR, 'test.png')})()
    
    content = model.prompt_to_content(MockPrompt())
    assert len(content) == 2  # Image block + text block
    assert 'image' in content[0]
    assert 'text' in content[1]
    assert content[1]['text'] == "Describe this image"

def test_prompt_to_content_multiple_attachments(model):
    """Test creating content with multiple attachments"""
    class MockPrompt:
        def __init__(self):
            self.prompt = "Describe these files"
            self.options = type('Options', (), {
                'bedrock_attach': f"{os.path.join(IMAGES_DIR, 'test.png')},{os.path.join(DOCS_DIR, 'test.txt')}"
            })()
    
    content = model.prompt_to_content(MockPrompt())
    assert len(content) == 3  # Image block + document block + text block
    assert 'image' in content[0]
    assert 'document' in content[1]
    assert 'text' in content[2]
    assert content[2]['text'] == "Describe these files"

def test_prompt_to_content_unsupported_file(model):
    """Test handling of unsupported file types"""
    class MockPrompt:
        def __init__(self):
            self.prompt = "Process this file"
            self.options = type('Options', (), {'bedrock_attach': 'test.unsupported'})()
    
    with pytest.raises(ValueError, match="Unable to guess mime type for file"):
        model.prompt_to_content(MockPrompt())

def test_encode_decode_bytes():
    """Test encoding and decoding of binary data"""
    model = BedrockClaude("test-model")
    test_data = {
        'image': {
            'format': 'png',
            'source': {
                'bytes': b'test binary data'
            }
        }
    }
    
    # Test encoding
    encoded = model.encode_bytes(test_data)
    assert 'bytes' not in encoded['image']['source']
    assert 'bytes_b64' in encoded['image']['source']
    
    # Test decoding
    decoded = model.decode_bytes(encoded)
    assert 'bytes_b64' not in decoded['image']['source']
    assert 'bytes' in decoded['image']['source']
    assert decoded['image']['source']['bytes'] == b'test binary data'
