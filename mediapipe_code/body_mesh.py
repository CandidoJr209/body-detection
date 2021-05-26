import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
drawing_spec_1 = mp_drawing.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=4)
drawing_spec_2 = mp_drawing.DrawingSpec(color=(255,255,2),thickness=2,circle_radius=2)

cap = cv2.VideoCapture(0)



with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.6) as holistic:

    while True:
    
        blank = np.zeros((480,640,3),dtype='uint8')

        _, frame = cap.read()
        
        frame = cv2.flip(frame,1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


        cv2.imshow('frame',image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

cap.release()
cv2.destroyAllWindows()