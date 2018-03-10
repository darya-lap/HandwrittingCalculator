import base64
from PIL import Image
import io as io

from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from matplotlib import pyplot as plt

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/postmethod', methods=['POST'])
def get_post_javascript_data():
    imgstring = request.form['javascript_data']
    imgstring = imgstring.split('base64,')[-1].strip()
    pic = io.StringIO()
    image_string = io.BytesIO(base64.b64decode(imgstring))
    image = Image.open(image_string)

    # Overlay on white background, see http://stackoverflow.com/a/7911663/1703216
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image, image)
    bg.save('C:/users/darya/desktop/pic.png')

    img = cv2.imread('C:/users/darya/desktop/pic.png', 0)

    img = cv2.medianBlur(img, 5)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(im2)
    print(contours)
    print(hierarchy)

    cv2.drawContours(img, contours, 3, (0, 255, 0), 3)



    plt.imshow(thresh, 'gray')
    plt.show()

    return imgstring


if __name__ == '__main__':
    app.run()
