import cv2
import random

cap = cv2.VideoCapture(0)

counters = [0, 0, 0, 0]

while True:


    ret, img = cap.read()

    if ret:
        cv2.imshow('img', img)

    k = cv2.waitKey(1)

    if k == ord('1'):
        cv2.imwrite('./data/train/compost/' + str(random.randint(0, 100000000)) + '.jpg', img)
        counters[1] += 1
        print(counters[1])

    elif k == ord('2'):
        cv2.imwrite('./data/train/recycle/' + str(random.randint(0, 100000000)) + '.jpg', img)
        counters[2] += 1
        print(counters[2])


    elif k == ord('3'):
        cv2.imwrite('./data/train/trash/' + str(random.randint(0, 100000000)) + '.jpg', img)
        counters[3] += 1
        print(counters[3])

    elif k == ord('0'):
        cv2.imwrite('./data/train/nothing/' + str(random.randint(0, 100000000)) + '.jpg', img)
        counters[0] += 1
        print(counters[0])

    elif k == ord('q'):
        print(counters)
        break
