import cv2
import numpy as np


def trainKNN():
    with np.load('alphabet_database.npz') as f:
        train = f['train']
        labels = f['label']
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, labels)

    return knn


def findContour(ar):
    ar = np.array(ar)

    max_xx = 0
    for x in ar:
        result = x[0] + x[2]
        if result > max_xx:
            max_xx = result

    max_yy = 0
    for y in ar:
        result = y[1] + y[3]
        if result > max_yy:
            max_yy = result

    x = np.min(ar[:, 0])
    w = max_xx - x
    y = np.min(ar[:, 1])
    h = max_yy - y

    return x, y, w, h


def cutRect(img):
    img_gray = img
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((30, 30), np.uint8)

    morph = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
    thr = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 5, 20)
    morph2 = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(
        morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = img.copy()
    result = [cv2.boundingRect(contour) for contour in contours]
    x, y, w, h = findContour(result)
    img_trim = img_copy[y:y+h, x:x+w]
    # cv2.imshow('cut', img_trim)
    return img_trim


def findCharacter(img):

    kernel = np.ones((8, 8), np.uint8)
    thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    bit = cv2.bitwise_not(thresh)
    dil = cv2.dilate(bit, kernel, iterations=3)

    contours = cv2.findContours(
        dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    img_copy = img.copy()
    cnts = [cv2.boundingRect(c) for c in contours]
    a = sorted(cnts)
    imgs = []
    for idx, contour in enumerate(a):
        x, y, w, h = contour[0], contour[1], contour[2], contour[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        img_trim = img_copy[y:y+h, x:x+w]
        # cv2.imshow('trim', img_trim)
        # cv2.waitKey(0)
        imgs.append(img_trim)

    return imgs


def mnistClassify(knn, imgs):
    results = []
    kernel = np.ones((3, 3), np.uint8)
    for target in imgs:
        target = cv2.bitwise_not(target)
        # target = cv2.dilate(target, kernel, iterations=1)
        target = cv2.resize(target, (28, 28), cv2.INTER_AREA)
        # cv2.imshow('mm', target)
        # cv2.waitKey(0)
        target = target.flatten().reshape((1, -1)).astype(np.float32)

        ret, result, neighbours, distance = knn.findNearest(target, k=111)
        results.append(result)
        print(neighbours)
    return results
