import numpy as np
import cv2

def check_is_movement(possible_objects):
    for i in possible_objects:
        if i[2] - i[0] > 30 or i[3] - i[1] > 30:
            return True

    if len(possible_objects) > 10:
        return True

    return False


def save_video(frame_array, frame, count):
    cv2.imwrite(home + '/video_results_3/'+str(count)+'.jpg', first_frame)
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter()
    vout.open(home + '/video_results_3/' + time.time().__str__() + '.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]),
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


cap = cv2.VideoCapture('/home/axis-inside/video_results_1/1522074958.9045799.mp4')

count = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
    	ret, frame = cap.read()

    if frame is None:
    	ret, frame = cap.read()

    if frame is None:
    	break
    
    print(count, frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    count += 1

cap.release()
cv2.destroyAllWindows()
