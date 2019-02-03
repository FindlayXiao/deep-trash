from keras.models import load_model
import numpy as np
import cv2
import serial

CATEGORY = ['nothing', 'compost', 'recycle', 'trash']
model = load_model('deeptrash.h5')
arduino = serial.Serial('COM3', 9600)

boundaries = [
    ('Orange', [17, 15, 100], [50, 56, 200]),
    ('Plastic', [86, 31, 4], [220, 88, 50]),
    ('Lays', [25, 146, 190], [62, 174, 250])
]


def predict(img):
    # if the size of the image is too small, report none
    if img.size < 100:
        return -1, 'none'

    # predict the image class
    pred = model.predict(img.reshape(1, *img.shape)[0:480, 80: 560])

    print(pred)

    # get the index of the maximum prediction value
    idx = np.argmax(pred)

    # return the index along with its associated category
    return idx, CATEGORY[idx]


def get_color_category(img):
    # loop over the boundaries
    for (name, lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow(name, output)

        denom = output.shape[0] * output.shape[1] * 255

        if name == 'Orange' and np.sum(output) / denom > 0.0:
            return 1

        elif name == 'Plastic' and np.sum(output) / denom > 0.0:
            return 2

        elif name == 'Lays' and np.sum(output) / denom > 0.0:
            return 3
        else:
            return 0


def run(show=True, prediction_threshold=5, crop=True):
    same_prediction_count = 0
    previous_prediction = None

    # open up the connection to the web cam
    cam = cv2.VideoCapture(0)

    # TODO: change to while
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

        # if q key is pressed, then quit
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


# TODO: implement the algorithm for dumping trash
def dump_trash(category):
    print('Running dump trash routine for', category, '. ' + CATEGORY[category])
    arduino.write(str(category).encode('utf-8'))


if __name__ == '__main__':
    run(show=True, crop=False)
