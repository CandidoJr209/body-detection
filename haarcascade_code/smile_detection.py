import cv2
import os

haarcascade_files_path = os.path.abspath(os.path.join( \
    os.path.dirname( __file__ ), '..', 'haarcascade_xmls')
    )
detection_face = cv2.CascadeClassifier(haarcascade_files_path + '/haarcascade_frontalface_default.xml')
detection_smile = cv2.CascadeClassifier(haarcascade_files_path + '/haarcascade_smile.xml')


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detection_face.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cropped_gray = gray[y:y+h, x:x+w]
        cropped_img = img[y:y+h, x:x+w]
        smiles = detection_smile.detectMultiScale(cropped_gray,1.2,6)

        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(cropped_img, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)


    cv2.imshow('smile detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()