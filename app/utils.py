import base64
import os

def save_base64_to_downloads(image_base64: str, filename: str = "generated_image.png") -> str:
    """
    Saves a base64-encoded image to the Downloads folder on macOS.

    Args:
        image_base64 (str): Base64 string of the image.
        filename (str): Desired filename for the saved image.

    Returns:
        str: Full path to the saved image.
    """
    downloads_path = "/Users/infinisync/Downloads"
    file_path = os.path.join(downloads_path, filename)

    # Decode the base64 string
    image_bytes = base64.b64decode(image_base64)

    # Save to file
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return file_path