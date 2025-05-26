from PIL.ImageFont import FreeTypeFont, ImageFont
from typing import Union


def getsize(font: Union[ImageFont, FreeTypeFont], text: str):
    left, top, right, bottom = font.getbbox(text)
    width = right - left
    height = bottom - top
    return width, height


def getheight(font: Union[ImageFont, FreeTypeFont], text: str):
    _, height = getsize(font, text)
    return height
