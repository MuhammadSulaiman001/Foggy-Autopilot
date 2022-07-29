import numpy as np
import cv2
from tensorflow.keras.models import load_model

from Autopilot.src import Utils


class AutopilotDriver:

    def __init__(self, wheelImg, carVideo, modelPath, smooth_angle):
        self.steer = cv2.imread(wheelImg, 0)
        self.cap = cv2.VideoCapture(carVideo)
        self.model = load_model(modelPath)
        self.smoothed_angle = smooth_angle

    def run(self):
        rows, cols = self.steer.shape
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
            steering_angle = Utils.keras_predict(self.model, gray)
            print(steering_angle)
            cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
            self.smoothed_angle += 0.2 * pow(abs((steering_angle - self.smoothed_angle)), 2.0 / 3.0) * (
                    steering_angle - self.smoothed_angle) / abs(
                steering_angle - self.smoothed_angle)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -self.smoothed_angle, 1)
            dst = cv2.warpAffine(self.steer, M, (cols, rows))
            cv2.imshow("steering wheel", dst)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    wheelImg = '../resources/steering_wheel_image.jpg'
    carVideo = '../resources/run.mp4'
    pathToFoggyModel = '../models/MsAutopilot_foggy.h5'
    pathToSunOnlyModel = '../models/MsAutopilot_sun_only.h5'
    modelPath = pathToFoggyModel
    smooth_angle = 0
    driver = AutopilotDriver(wheelImg, carVideo, modelPath, smooth_angle)
    # Press q to exit openCV..
    driver.run()
