import numpy as np
import cv2
import time
import os
import nms
import traceback

home = os.path.expanduser('~')


def check_is_movement(possible_objects):
    for i in possible_objects:
        if i[2] - i[0] > 30 or i[3] - i[1] > 30:
            return True

    if len(possible_objects) > 10:
        return True

    return False


def save_video(frame_array, frame):
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter()
    vout.open(home + '/' + time.time().__str__() + '.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]),
                        True)
    for frame in frame_array:
        vout.write(frame)

    vout.release()
    cv2.destroyAllWindows()


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


def pad_black(frame):
    frame[0:2] = np.full(frame[0:2].shape, fill_value=0, dtype=np.uint8)
    frame[-2:] = np.full(frame[-2:].shape, fill_value=0, dtype=np.uint8)
    frame[:, 0:2] = np.full(frame[:, 0:2].shape, fill_value=0, dtype=np.uint8)
    frame[:, -2:] = np.full(frame[:, -2:].shape, fill_value=0, dtype=np.uint8)

    return frame


def check_with_first_frame(first_frame, intermediate_frame):
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    intermediate_frame_gray = cv2.cvtColor(intermediate_frame, cv2.COLOR_BGR2GRAY)
    frameDelta = cv2.absdiff(first_frame_gray, intermediate_frame_gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, ksize=7)
    thresh = pad_black(thresh)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    possible_objects = apply_contours(dilated)
    return check_is_movement(possible_objects)

cap = cv2.VideoCapture('/Users/sumithkrishna/Downloads/very dangerous ATM cctv footage in INDIA.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
cv2.createBackgroundSubtractorMOG2(detectShadows=False)

kernel = np.ones((5, 5), np.uint8)
frame_count = 1
previous = 0

first_frame = None
frame_array = []
no_mov_arr = []
rejected_arr = []
while(1):
    ret, frame = cap.read()
    if frame is None:
        if previous == 0:
            previous = 1
            continue
        if previous == 1:
            break

    if first_frame is None:
       first_frame = frame
    try:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, ksize=7)
        fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        fgmask = pad_black(fgmask)
        possible_objects = apply_contours(fgmask)
        possible_objects = nms.non_max_suppression(np.asarray(possible_objects))
        frame_count += 1

        if check_is_movement(possible_objects):
              frame_array.append(frame)

        elif check_with_first_frame(first_frame, frame):
            frame_array.append(frame)

            if len(rejected_arr) == 0:
                rejected_arr.append(frame_count)

            if frame_count - rejected_arr[-1] >1:
                rejected_arr = [frame_count]

            if frame_count - rejected_arr[-1] == 1:
                rejected_arr.append(frame_count)

            if len(rejected_arr) >=12:
                save_video(list(frame_array), frame)
                frame_array =[]
                first_frame = frame
                rejected_arr = []

        else:
            if len(frame_array)!=0:
                if len(no_mov_arr) == 0:
                    no_mov_arr.append(frame_count)

                if frame_count - no_mov_arr[-1] >1:
                    no_mov_arr = [frame_count]

                if frame_count - no_mov_arr[-1] == 1:
                    no_mov_arr.append(frame_count)

                if len(no_mov_arr) >=12:
                    save_video(list(frame_array), frame)
                    frame_array =[]
                    first_frame = frame
                    no_mov_arr = []

        print(frame_count, frame.shape, "No Mov Arr len ", len(no_mov_arr))
        # cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    except Exception:
        traceback.print_exc()
        break
cap.release()
cv2.destroyAllWindows()