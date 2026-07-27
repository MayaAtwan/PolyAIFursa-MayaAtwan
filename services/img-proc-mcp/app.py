import base64
import io
from fastmcp import FastMCP
from PIL import Image, ImageFilter
import numpy as np

mcp = FastMCP("img-proc")


def _decode(b64: str) -> Image.Image:
    b64 += "=" * (-len(b64) % 4)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@mcp.tool()
def rotate(degrees: float, image_b64: str = "") -> str:
    """Rotate the image counter-clockwise by the given degrees. image_b64 is supplied automatically by the agent."""
    return _encode(_decode(image_b64).rotate(degrees, expand=True))


@mcp.tool()
def flip(direction: str, image_b64: str = "") -> str:
    """Flip the image. direction must be 'horizontal' or 'vertical'. image_b64 is supplied automatically by the agent."""
    img = _decode(image_b64)
    ops = {"horizontal": Image.FLIP_LEFT_RIGHT, "vertical": Image.FLIP_TOP_BOTTOM}
    if direction not in ops:
        raise ValueError(f"Unknown direction '{direction}'. Use 'horizontal' or 'vertical'.")
    return _encode(img.transpose(ops[direction]))


@mcp.tool()
def blur(radius: float = 2.0, image_b64: str = "") -> str:
    """Apply Gaussian blur to the image. image_b64 is supplied automatically by the agent."""
    return _encode(_decode(image_b64).filter(ImageFilter.GaussianBlur(radius)))


@mcp.tool()
def resize(width: int, height: int, image_b64: str = "") -> str:
    """Resize the image to width x height pixels. image_b64 is supplied automatically by the agent."""
    return _encode(_decode(image_b64).resize((width, height), Image.LANCZOS))


@mcp.tool()
def crop(x1: int, y1: int, x2: int, y2: int, image_b64: str = "") -> str:
    """Crop the image to the bounding box [x1, y1, x2, y2]. image_b64 is supplied automatically by the agent."""
    return _encode(_decode(image_b64).crop((x1, y1, x2, y2)))


@mcp.tool()
def add_noise(amount: float = 0.05, image_b64: str = "") -> str:
    """Add salt-and-pepper noise. amount is the fraction of pixels to corrupt. image_b64 is supplied automatically by the agent."""
    img = _decode(image_b64)
    arr = np.array(img)
    n = int(amount * arr.shape[0] * arr.shape[1])
    for _ in range(n):
        arr[np.random.randint(arr.shape[0]), np.random.randint(arr.shape[1])] = 255
    for _ in range(n):
        arr[np.random.randint(arr.shape[0]), np.random.randint(arr.shape[1])] = 0
    return _encode(Image.fromarray(arr.astype(np.uint8)))


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=9000)
