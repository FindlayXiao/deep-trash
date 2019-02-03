import time

from keras.models import load_model
import numpy as np
import cv2
import serial
import time

CATEGORY = ['nothing', 'compost', 'recycle', 'trash']
model = load_model('deeptrash.h5')
arduino = serial.Serial('COM3', 9600)
fgbg = cv2.createBackgroundSubtractorMOG2()

boundaries = [
    ('Orange', [17, 15, 100], [50, 56, 200]),
    # ('Plastic', [25, 146, 190], [62, 174, 250]),
    ('Trash', [86, 31, 4], [255, 88, 50])
]


def predict(img):
    # if the size of the image is too small, report none
    if img.size < 100:
        return -1, 'none'

    # predict the image class
    pred = model.predict(img.reshape(1, *img.shape)[0:480, 80: 560])

    # get the index of the maximum prediction value
    idx = np.argmax(pred)

    # return the index along with its associated category
    return idx, CATEGORY[idx]


def get_background_score(img):
    denom = img.shape[0] * img.shape[1] * img.shape[2]
    mask = fgbg.apply(img) / 255
    return np.sum(mask) / denom


def get_color_category(img):
    # loop over the boundaries
    for (name, lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        denom = output.shape[0] * output.shape[1] * 255

        if name == 'Orange' and np.sum(output) / denom > 0.00001:
            return 1

        if name == 'Trash' and np.sum(output) / denom > 0.0001:
            return 3


def run(show=True, prediction_threshold=5, crop=True):
    same_prediction_count = 0
    previous_prediction = None

    # open up the connection to the web cam
    cam = cv2.VideoCapture(0)

    states = ['Compost', 'Recyclable', 'Trash']
    curr_state = 0
    frame_buffer = 60

    while True:
        # capture the web cam image
        ret, img = cam.read()

        # if an image was retrived successfully
        if ret:

            cv2.imshow('img', img)

            # make a prediction based on the cropped image
            # idx, category = predict(img) if get_background_score(img, learning_rate=0.001) > 0.1 else (-1, 'none')
            idx, category = predict(img)
            c_category = get_color_category(img)

            # if q key is pressed, then quit
            if cv2.waitKey(1) == 27:
                break

            score = get_background_score(img)
            if score > 0.075 and frame_buffer <= 0:
                frame_buffer = 60
                print(states[curr_state % 3] + ' was detected.')
                dump_trash(curr_state)
                curr_state += 1

            frame_buffer -= 1
            continue

            if c_category == 1:
                print('Compost detected!')
                # dump_trash(c_category)
            elif c_category == 2:
                print('Recyclable item detected!')
            else:
                continue

            # if the previous state is the same as the current state
            if previous_prediction == idx and idx != -1:
                # add one to the counter
                same_prediction_count += 1
            else:
                # otherwise, set the counter to 0
                same_prediction_count = 0

            # if the prediction count reaches the threshold
            if same_prediction_count >= prediction_threshold:
                # dump the trash
                dump_trash(idx)

            # set the previous prediction to the current prediction
            previous_prediction = idx



    cv2.destroyAllWindows()


# TODO: implement the algorithm for dumping trash
def dump_trash(category):
    # print('Running dump trash routine for', category, CATEGORY[category % 3])
    time.sleep(0.5)
    arduino.write(str((category - 1) % 3).encode('utf-8'))
    time.sleep(1.5)


if __name__ == '__main__':
    run(show=True, crop=False)
