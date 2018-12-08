# coding:utf-8
import cv2
import sys
reload(sys)


sys.setdefaultencoding('utf8')

# cap = cv2.VideoCapture(u'http://192.168.43.1:8081')

cap = cv2.VideoCapture(1)

while True:
    ret,frame = cap.read()
    print(ret,frame.shape)
    if ret == True:
        cv2.imshow("frame", frame)
        cv2.waitKey(10)
    else:
        break
cap.release()

cv2.destroyAllWindows()

