import numpy as np
import cv2
from PIL import Image
import pytesseract
import argparse
import os


def main():
    img = cv2.imread('C:/users/darya/desktop/pic.png')
    mser = cv2.MSER_create()

    # img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    # img = img[5:-5, 5:-5, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # vis = img.copy()

    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    i = 0
    hulls1 = []
    bounds = []
    begin = 0
    # end = 0

    xminp = np.min(hulls[0], axis=0)[0][0]
    # yminp = np.min(hulls[0], axis=0)[0][1]
    xmaxp = np.max(hulls[0], axis=0)[0][0]
    # ymaxp = np.max(hulls[0], axis=0)[0][1]
    for inner in hulls:
        xmin = np.min(inner, axis=0)[0][0]
        ymin = np.min(inner, axis=0)[0][1]
        xmax = np.max(inner, axis=0)[0][0]
        ymax = np.max(inner, axis=0)[0][1]
        for outer in hulls:
            if (xmin > np.min(outer, axis=0)[0][0]) and (xmax < np.max(outer, axis=0)[0][0]) and (
                    ymin > np.min(outer, axis=0)[0][1]) and (ymax < np.max(outer, axis=0)[0][1]):
                break
        else:
            if xmax < xminp:
                bounds.append([begin, i - 1])
                break
            hulls1.append(inner)
            i = i + 1
            if xmin > xmaxp:
                bounds.append([begin, i - 1])
                begin = i
            xminp = xmin
            xmaxp = xmax
            # yminp = ymin
            # ymaxp = ymax
    print(bounds)
    hulls = hulls1

    images_parts = []
    for bound in bounds:
        # print(bound[0], bound[1])
        # print(hulls[bound[0]:bound[1]])
        images_parts.append(hulls[bound[0]:bound[1]])

    i = 1
    for part in images_parts:
        minx = min([x[0][0] for x in part[0]])
        maxx = max([x[0][0] for x in part[0]])
        miny = min([y[0][1] for y in part[0]])
        maxy = max([y[0][1] for y in part[0]])
        width = maxx - minx
        height = maxy - miny
        part_of_image = img[miny:maxy, minx:maxx]
        print('uuuuu')
        print(part_of_image.shape)

        if height > width:
            diff = height-width
            background = np.full((height, diff//2, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((part_of_image, background), axis=1)
            background = np.full((height, diff-diff//2, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((background, part_of_image), axis=1)
        if width > height:
            diff = width-height
            background = np.full((diff//2, width, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((part_of_image, background), axis=0)
            background = np.full((diff-diff//2, width, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((background, part_of_image), axis=0)

        filename = 'C:/users/darya/desktop/pic' + str(i) + '.png'
        i = i + 1
        cv2.imwrite(filename, part_of_image)
        # cv2.namedWindow('img', 0)
        # cv2.polylines(vis, hulls, 1, (0, 255, 0))
    # while cv2.waitKey() != ord('q'):
    #     continue
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
