import cv2 as cv
import numpy as np


class VolleyballTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)

        
        self.haar_cascade = cv.CascadeClassifier("haar_full_body.xml")
        if self.haar_cascade.empty():
            raise IOError("Failed to load Haar cascade XML.")

        
        self.kernel = np.ones((3, 3), np.uint8)
        self.lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        self.upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, ball_center = self.detect_ball(frame)
            frame, team1_count, team2_count = self.detect_players(frame)

            
            cv.putText(frame, f'Team 1: {team1_count}', (50, 50),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Team 2: {team2_count}', (50, 80),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            
            cv.imshow('Volleyball Tracking', frame)

            if cv.waitKey(1) & 0xFF == ord('l'):
                break

        self.cap.release()
        cv.destroyAllWindows()

    def detect_ball(self, frame):
        blurred = cv.GaussianBlur(frame, (7, 7), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
        mask = cv.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ball_center = None

        for contour in contours:
            area = cv.contourArea(contour)
            if 100 < area < 800:
                (x, y), radius = cv.minEnclosingCircle(contour)
                circularity = 4 * np.pi * area / (cv.arcLength(contour, True) ** 2 + 1e-6)
                if 0.7 < circularity <= 1.2 and 5 < radius < 30:
                    ball_center = (int(x), int(y))
                    cv.circle(frame, ball_center, int(radius), (0, 255, 0), 2)

        return frame, ball_center

    def detect_players(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.rectangle(mask, (100, 150), (1340, 700), 255, -1)
        masked_frame = cv.bitwise_and(frame, frame, mask=mask)
        gray = cv.cvtColor(masked_frame, cv.COLOR_BGR2GRAY)

        players = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

        for (x, y, w, h) in players:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

        
        return frame, len(players) // 2, len(players) // 2


if __name__ == "__main__":
    tracker = VolleyballTracker('volleyball_match.mp4')
    tracker.run()
