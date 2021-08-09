import cv2
import numpy as np
from Font_utils import *
from PIL import ImageFont, ImageDraw, Image


key = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: "K", 11: 'L', 12: 'M', 13: "N", 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}


def onChange(pos):
    return 'CHECK', pos


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

knn = trainKNN()

shot = False
done = True
print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

while(True):
    # ret, frame = cap.read()
    if not done:
        cv2.destroyAllWindows()
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        ret, frame = cap.read()
        done = True
    else:
        ret, frame = cap.read()

    if(ret):
        if cv2.waitKey(1) == ord('q'):
            break
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)

        cv2.imshow('Trackbar test', frame)

        if cv2.waitKey(1) == ord("s"):
            shot = True
            cv2.destroyAllWindows()
            cv2.imwrite('shot.jpg', frame)
            cap.release()
            img = cv2.imread('shot.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cut_img = cutRect(img)
            cut_imgs = findCharacter(cut_img)
            fonts = mnistClassify(knn, cut_imgs)
            img_basic = cv2.resize(img, (640, 480))
            text = ''
            for i in fonts:
                text += key[i[0][0]]
            arr = np.zeros((480, 640), np.uint8)
            cv2.namedWindow('Trackbar test')
            cv2.createTrackbar('Fonts', 'Trackbar test', 0, 10, onChange)
            track_img = np.concatenate((img_basic, arr), axis=1)
            cv2.imshow('Trackbar test', track_img)
            while shot:
                font_number = cv2.getTrackbarPos('Fonts', 'Trackbar test')
                fontpath = f"fonts/{font_number}.ttf"
                font = ImageFont.truetype(fontpath, 80)
                img_pil = Image.fromarray(arr)
                draw = ImageDraw.Draw(img_pil)
                draw.text((100, 150),  text, fill='white', font=font)
                area = np.array(img_pil)
                track_img = np.concatenate((img_basic, area), axis=1)
                keycode = cv2.waitKey()
                cv2.imshow('Trackbar test', track_img)
                if keycode == ord('z'):
                    shot = False
                    done = False

cv2.destroyAllWindows()
