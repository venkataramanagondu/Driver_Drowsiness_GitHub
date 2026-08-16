import cv2
import time
import winsound
import threading
import geocoder
import mediapipe as mp
import numpy as np
import pygame
import wave
import os
from scipy.spatial import distance
from dotenv import load_dotenv
from twilio.rest import Client


# ================= ENVIRONMENT =================
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_RECEIVER_NUMBER = os.getenv("TWILIO_RECEIVER_NUMBER")

if not all([
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    TWILIO_RECEIVER_NUMBER
]):
    raise ValueError("Twilio credentials not found in .env file.")

client = Client(
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN
)


# ================= VIBRATION =================
VIBRATION_FILE = "vibration.wav"


def generate_vibration_wav():

    sample_rate = 44100
    duration = 2.0
    freq = 45

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration)
    )

    envelope = 0.5 * (
        1 + np.sign(
            np.sin(2 * np.pi * 8 * t)
        )
    )

    wave_data = (
        np.sin(2 * np.pi * freq * t)
        * envelope
        * 32767
    ).astype(np.int16)

    with wave.open(VIBRATION_FILE, "w") as f:

        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)

        f.writeframes(
            wave_data.tobytes()
        )


if not os.path.exists(VIBRATION_FILE):

    generate_vibration_wav()

    print("Vibration audio generated.")


# ================= PYGAME =================
pygame.mixer.init(
    frequency=44100,
    size=-16,
    channels=1,
    buffer=512
)

vibration_sound = pygame.mixer.Sound(
    VIBRATION_FILE
)

vibration_sound.set_volume(1.0)


# ================= MEDIAPIPE =================
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarkerOptions,
    FaceLandmarker
)


MODEL_PATH = "face_landmarker.task"


if not os.path.exists(MODEL_PATH):

    import urllib.request

    print("Downloading face landmarker model...")

    urllib.request.urlretrieve(
        "https://storage.googleapis.com/"
        "mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/"
        "face_landmarker.task",
        MODEL_PATH
    )

    print("Model downloaded.")


options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    num_faces=1
)

detector = FaceLandmarker.create_from_options(
    options
)


# ================= EYE LANDMARKS =================
LEFT_EYE = [
    362, 385, 387,
    263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158,
    133, 153, 144
]

EAR_THRESHOLD = 0.22
EAR_FRAMES = 8


# ================= STATE =================
speed = 0.0
engine_on = False
system_active = False

eye_closed_start = None
closed_frames = 0

sms_sent = False


# ================= ALARM =================
stop_flag = threading.Event()
alarm_thread = None


def alarm_loop():

    while not stop_flag.is_set():

        winsound.Beep(3500, 20)

        if stop_flag.is_set():
            break

        winsound.Beep(2000, 20)


def start_alarm():

    global alarm_thread

    if alarm_thread and alarm_thread.is_alive():
        return

    stop_flag.clear()

    alarm_thread = threading.Thread(
        target=alarm_loop,
        daemon=True
    )

    alarm_thread.start()


# ================= VIBRATION =================
def start_vibration():

    if not pygame.mixer.get_busy():

        vibration_sound.play(
            loops=-1
        )


def stop_vibration():

    vibration_sound.stop()


def stop_all_alerts():

    stop_flag.set()
    stop_vibration()


# ================= EAR CALCULATION =================
def ear_calc(lm, eye, w, h):

    pts = [
        (
            int(lm[i].x * w),
            int(lm[i].y * h)
        )
        for i in eye
    ]

    A = distance.euclidean(
        pts[1],
        pts[5]
    )

    B = distance.euclidean(
        pts[2],
        pts[4]
    )

    C = distance.euclidean(
        pts[0],
        pts[3]
    )

    return (
        (A + B) / (2.0 * C),
        pts
    )


# ================= TWILIO =================
def send_sms(location):

    body = (
        "EMERGENCY ALERT\n\n"
        "Driver drowsiness detected!\n"
        "Vehicle has been stopped.\n\n"
        f"Location:\n{location}"
    )

    try:

        message = client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=TWILIO_RECEIVER_NUMBER
        )

        print(
            "Vehicle stopped. "
            "SMS sent successfully."
        )

        print("Message SID:", message.sid)

        return True

    except Exception as e:

        print("Twilio error:")
        print(e)

        return False


# ================= MAIN LOOP =================
cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_BRIGHTNESS,
    150
)

print(
    "Press 'S' to Start Vehicle | ESC to Exit"
)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    key = cv2.waitKey(1) & 0xFF


    # START
    if key == ord("s"):

        system_active = True
        engine_on = True

        print(
            "System Started — "
            "Monitoring Driver..."
        )


    # EXIT
    if key == 27:

        stop_all_alerts()

        break


    # WAIT
    if not system_active:

        cv2.putText(
            frame,
            "Press S to Start",
            (160, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "Driver Drowsiness System",
            frame
        )

        continue


    # FACE DETECTION
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_img)

    eyes_closed = False


    if result.face_landmarks:

        lm = result.face_landmarks[0]

        l_ear, l_pts = ear_calc(
            lm,
            LEFT_EYE,
            w,
            h
        )

        r_ear, r_pts = ear_calc(
            lm,
            RIGHT_EYE,
            w,
            h
        )

        avg_ear = (
            l_ear + r_ear
        ) / 2.0


        for pt in l_pts + r_pts:

            cv2.circle(
                frame,
                pt,
                2,
                (0, 255, 0),
                -1
            )


        cv2.polylines(
            frame,
            [np.array(l_pts)],
            True,
            (0, 255, 0),
            1
        )

        cv2.polylines(
            frame,
            [np.array(r_pts)],
            True,
            (0, 255, 0),
            1
        )


        cv2.putText(
            frame,
            f"EAR: {avg_ear:.2f}",
            (w - 160, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )


        if avg_ear < EAR_THRESHOLD:

            closed_frames += 1

        else:

            closed_frames = 0


        eyes_closed = (
            closed_frames >= EAR_FRAMES
        )


        if eyes_closed:

            cv2.putText(
                frame,
                "EYES CLOSED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "EYES OPEN",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    else:

        closed_frames = 0

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 100, 255),
            2
        )


    # ACCELERATION
    if (
        engine_on
        and not eyes_closed
        and speed < 70
    ):

        speed += 0.25


    # DROWSINESS STAGES
    if eyes_closed:

        if eye_closed_start is None:

            eye_closed_start = time.time()


        elapsed = (
            time.time()
            - eye_closed_start
        )


        # STAGE 1
        if elapsed >= 1:

            cv2.putText(
                frame,
                "!! DROWSY ALERT !!",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            start_alarm()


        # STAGE 2
        if elapsed >= 2:

            cv2.putText(
                frame,
                "SEAT VIBRATION ON",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 220, 255),
                2
            )

            start_vibration()


        # STAGE 3
        if (
            elapsed >= 3
            and speed > 0
        ):

            speed = max(
                0,
                speed - 0.467
            )

            cv2.putText(
                frame,
                "AUTO BRAKING ACTIVE",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 80, 255),
                2
            )


        # STAGE 4
        if (
            speed <= 0
            and not sms_sent
        ):

            stop_all_alerts()

            g = geocoder.ip("me")

            if g.latlng:

                location = (
                    f"https://maps.google.com/"
                    f"?q={g.latlng[0]},{g.latlng[1]}"
                )

            else:

                location = "Location unavailable"


            send_sms(location)

            sms_sent = True

            break


    else:

        if eye_closed_start is not None:

            eye_closed_start = None

            stop_all_alerts()

            sms_sent = False


    # SPEED
    cv2.putText(
        frame,
        f"Speed: {int(speed)} km/h",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Driver Drowsiness System",
        frame
    )


# ================= CLEANUP =================
cap.release()

cv2.destroyAllWindows()

detector.close()

pygame.mixer.quit()