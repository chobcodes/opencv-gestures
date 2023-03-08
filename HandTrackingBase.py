import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, static_mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, track_confidence=0.5):
        self.mode = static_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.d_conf = detection_confidence
        self.t_conf = track_confidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity, self.d_conf, self.t_conf)

    def findHands(self, img, draw=True):
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPos(self, img, handNum=0, draw=True):

        landmarkList = []
        if self.results.multi_hand_landmarks:
            selectedHand = self.results.multi_hand_landmarks[handNum]

            for id, landmark in enumerate(selectedHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarkList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (0,255,255), cv2.FILLED)

        return landmarkList

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        landmarkList = detector.findPos(img)
        if len(landmarkList):
            print(landmarkList[4])

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()

