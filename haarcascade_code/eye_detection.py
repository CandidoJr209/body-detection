import cv2
import os

haarcascade_files_path = os.path.abspath(os.path.join( \
    os.path.dirname( __file__ ), '..', 'haarcascade_xmls')
    )

detection_face = cv2.CascadeClassifier( \
    haarcascade_files_path + '/haarcascade_frontalface_default.xml' 
    )
detection_eye = cv2.CascadeClassifier( \
    haarcascade_files_path + '/haarcascade_eye.xml'
    )


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detection_face.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cropped_gray = gray[y:y+h, x:x+w]
        cropped_img = img[y:y+h, x:x+w]
        eyes = detection_eye.detectMultiScale(cropped_gray,1.2,7)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(cropped_img, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)


    cv2.imshow('eye detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()