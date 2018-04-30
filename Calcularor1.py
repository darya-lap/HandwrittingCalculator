from PIL import Image
import io as io
import functionality.cutting
import cv2
import numpy as np


from flask import Flask, render_template, request
import base64

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/postmethod', methods=['POST'])
def get_post_javascript_data():
    imgstring = request.form['javascript_data']
    imgstring = imgstring.split('base64,')[-1].strip()
    image_string = io.BytesIO(base64.b64decode(imgstring))
    image = Image.open(image_string)

    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image, image)
    image_array = np.array(bg)
    #bg.save('C:/users/darya/desktop/pic.png')

    parts_of_image = functionality.cutting.cut(image_array)

    #with open('../entry.pickle','wb') as f:
        #pickle.dump(img, f)

    return imgstring


if __name__ == '__main__':
    app.run()
