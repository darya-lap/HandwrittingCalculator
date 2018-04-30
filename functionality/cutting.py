import numpy as np
import cv2


def cut(img):
    mser = cv2.MSER_create()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    i = 0
    hulls1 = []
    bounds = []
    begin = 0

    xminp = np.min(hulls[0], axis=0)[0][0]
    xmaxp = np.max(hulls[0], axis=0)[0][0]

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
    hulls = hulls1

    #i = 1
    parts_of_image = []
    for bound in bounds:
        part = hulls[bound[0]:bound[1]]
        minx = min([x[0][0] for x in part[0]])
        maxx = max([x[0][0] for x in part[0]])
        miny = min([y[0][1] for y in part[0]])
        maxy = max([y[0][1] for y in part[0]])
        width = maxx - minx
        height = maxy - miny
        part_of_image = img[miny:maxy, minx:maxx]

        if height > width:
            diff = height - width
            background = np.full((height, diff // 2, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((part_of_image, background), axis=1)
            background = np.full((height, diff - diff // 2, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((background, part_of_image), axis=1)
        if width > height:
            diff = width - height
            background = np.full((diff // 2, width, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((part_of_image, background), axis=0)
            background = np.full((diff - diff // 2, width, 3), 255, dtype=np.uint8)
            part_of_image = np.concatenate((background, part_of_image), axis=0)

        parts_of_image.append(part_of_image)
        return parts_of_image
