import base64
import re

import cv2.cv2 as cv2
import numpy as np
from bs4 import BeautifulSoup


class EmbedImageCompressor(object):
    _pattern = 'data:image/\w+;base64,(.+)'
    _pattern_compiled = re.compile(_pattern, re.S)
    _src_prefix = 'data:image/jpeg;base64,'

    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.content = f.read()
        self.soup = BeautifulSoup(self.content, 'html.parser')

    def process(self):
        for tag in self.soup.findAll('img'):
            src_str = tag.attrs.get('src')
            result = self._pattern_compiled.match(src_str)
            if result is None:
                continue

            img_b64 = result.group(1)
            img_cv2 = decode(img_b64)
            img_b64_new = compress_and_encode(img_cv2)

            tag_new = self.soup.new_tag('img')
            tag_new.attrs['src'] = f'{self._src_prefix}{img_b64_new}'
            tag.replace_with(tag_new)

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.soup.prettify())


def compress_and_encode(img: np.ndarray, quality=95) -> str:
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, img_compressed = cv2.imencode('.jpg', img, encode_param)
    # TODO: handle fail of encoding appropriately
    if not success:
        raise ValueError

    img_b64 = base64.b64encode(img_compressed)
    return img_b64.decode('utf-8')


def decode(img_b64: str) -> np.ndarray:
    img_data = base64.b64decode(img_b64)
    img_np = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_ANYCOLOR)
    return img
