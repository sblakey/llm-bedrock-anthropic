from PIL import Image
import os

# Create test images directory if it doesn't exist
images_dir = os.path.join('tests', 'fixtures', 'images')
os.makedirs(images_dir, exist_ok=True)

# Create test PNG image
png_image = Image.new('RGB', (100, 100), color='red')
png_image.save(os.path.join(images_dir, 'test.png'))

# Create test JPEG image
jpeg_image = Image.new('RGB', (100, 100), color='blue')
jpeg_image.save(os.path.join(images_dir, 'test.jpg'))

# Create test GIF image
gif_image = Image.new('RGB', (100, 100), color='green')
gif_image.save(os.path.join(images_dir, 'test.gif'))

# Create test WEBP image
webp_image = Image.new('RGB', (100, 100), color='yellow')
webp_image.save(os.path.join(images_dir, 'test.webp'))

# Create large test image (over max size)
large_image = Image.new('RGB', (2000, 2000), color='purple')
large_image.save(os.path.join(images_dir, 'large_test.png'))

# Create test documents directory if it doesn't exist
docs_dir = os.path.join('tests', 'fixtures', 'documents')
os.makedirs(docs_dir, exist_ok=True)

# Create test text file
with open(os.path.join(docs_dir, 'test.txt'), 'w') as f:
    f.write('This is a test document.')

# Create test markdown file
with open(os.path.join(docs_dir, 'test.md'), 'w') as f:
    f.write('# Test Markdown\nThis is a test markdown document.')

print("Test fixtures created successfully!")
