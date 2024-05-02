import cv2
import mediapipe as mp

def main():
    # Load MediaPipe gesture recognizer
    mp_holistic = mp.solutions.holistic
    gesture_recognizer = mp_holistic.Holistic(static_image_mode=False, model_path="gesture_recognizer.task")

    # Open webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect gestures
        results = gesture_recognizer.process(rgb_frame)

        # Visualize the gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
