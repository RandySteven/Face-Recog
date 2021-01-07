import cv2
cam = cv2.VideoCapture(0)

while True:
    retV, frame = cam.read()
    # abuAbu = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Video", frame)
    # cv2.imshow("Video 2", abuAbu)
    k = cv2.waitKey(1) & 0xFF 
    if k == 27 or k == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()