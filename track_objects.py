import numpy as np
import cv2
import time
import os
import nms


home = os.path.expanduser('~')


def apply_contours(image: np.ndarray):
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []

    for c in contours:
        shp = np.asarray(c).shape

        if shp[0] is not 1 or shp[1] is not 1:
            new_contours.append(c)

    rect_list = []
    for contour in contours[1:]:
        arr = np.squeeze(np.asarray(contour), axis=1)
        rect_list.append([min(arr[:, 0:-1])[0], min(arr[:, 1:])[0], max(arr[:, 0:-1])[0], max(arr[:, 1:])[0]])

    return rect_list


def remove_noise(img):
    morph = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    morph = cv2.erode(morph, kernel1, iterations=1)

    return morph

cap = cv2.VideoCapture('/Users/sumithkrishna/Downloads/very dangerous ATM cctv footage in INDIA.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
cv2.createBackgroundSubtractorMOG2(detectShadows=False)

fps = 25

_, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout = cv2.VideoWriter()
success = vout.open(home+'/'+time.time().__str__()+'.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]), True)
kernel = np.ones((5,5),np.uint8)
count = 1
previous = 0
while(1):
    ret, frame = cap.read()
    if frame is None:
        if previous == 0:
            previous = 1
            continue
        if previous == 1:
            break
    try:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, ksize=7)
        fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        possible_objects = apply_contours(fgmask)
        possible_objects = nms.non_max_suppression(np.asarray(possible_objects))
        for r in possible_objects:
          cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 3)
        # cv2.imwrite("masked_images_1/"+str(count)+'.jpg', frame)
        count += 1
        vout.write(frame)
        print(count, frame.shape)
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    except Exception:
        break
vout.release()
cap.release()
cv2.destroyAllWindows()