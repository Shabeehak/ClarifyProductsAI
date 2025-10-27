import pytesseract
from PIL import Image

# Optional: explicitly set the Tesseract path (use this if Python can't find it)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test on an image
text = pytesseract.image_to_string(Image.open("sample_image.jpg"))
print(text)
