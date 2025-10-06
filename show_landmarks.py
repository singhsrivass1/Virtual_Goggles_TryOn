import cv2, mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        h, w = frame.shape[:2]
        for lm in res.multi_face_landmarks[0].landmark:
            x, y = int(lm.x*w), int(lm.y*h)
            cv2.circle(frame,(x,y),1,(0,255,0),-1)
    cv2.imshow("Landmarks (press q to quit)",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):break
cap.release();cv2.destroyAllWindows()
