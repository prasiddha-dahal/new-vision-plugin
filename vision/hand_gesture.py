import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks

MODEL_PATH    = "hand_landmarker.task"
PHOTO_DIR     = "/home/prasiddha/Pictures/GesturesImages"

GESTURE_COOLDOWN = 1.5
MOVE_COOLDOWN    = 0.016

_landmarker: HandLandmarker | None = None


def _get_landmarker() -> HandLandmarker:
    global _landmarker
    if _landmarker is None:
        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options      = HandLandmarkerOptions(
            base_options                  = base_options,
            running_mode                  = RunningMode.IMAGE,
            num_hands                     = 2,
            min_hand_detection_confidence = 0.75,
            min_hand_presence_confidence  = 0.75,
            min_tracking_confidence       = 0.75
        )
        _landmarker = HandLandmarker.create_from_options(options)
    return _landmarker


def detect_gesture(lm) -> str:
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y

    if index_up and not any([middle_up, ring_up, pinky_up]):
        return "POINT_LEFT" if lm[8].x < lm[5].x else "POINT_RIGHT"

    if index_up and middle_up and not any([ring_up, pinky_up]):
        return "V_SIGN"

    if all([index_up, middle_up, ring_up, pinky_up]):
        return "OPEN_HAND"

    if not any([index_up, middle_up, ring_up, pinky_up]):
        return "FIST"

    return "NONE"


def detect_from_jpeg(
    jpeg_bytes    : bytes,
    current_time  : float,
    prev_palm     : tuple | None,
    prev_gesture  : str,
    last_trigger_time : float,
    last_move_time    : float,
) -> tuple[dict | None, tuple | None, str, float, float]:

    arr   = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return None, prev_palm, prev_gesture, last_trigger_time, last_move_time

    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data         = imageRGB
    )

    landmarker = _get_landmarker()
    result     = landmarker.detect(mp_image)
    all_hands  = result.hand_landmarks or []

    can_trigger = current_time - last_trigger_time > GESTURE_COOLDOWN
    can_move    = current_time - last_move_time    > MOVE_COOLDOWN

    event: dict | None = None

    if all_hands:
        lm      = all_hands[0]
        gesture = detect_gesture(lm)
        palm    = (lm[9].x, lm[9].y)

        if gesture == "FIST":
            if prev_palm is not None and can_move:
                dx = palm[0] - prev_palm[0]
                dy = palm[1] - prev_palm[1]

                if abs(dx) > 0.025 or abs(dy) > 0.025:
                    direction      = ("r" if dx > 0 else "l") if abs(dx) > abs(dy) else ("d" if dy > 0 else "u")
                    last_move_time = current_time
                    event          = {
                        "gesture"    : "FIST",
                        "hand_count" : len(all_hands),
                        "confidence" : 0.0,
                        "palm"       : {"x": palm[0], "y": palm[1]},
                        "timestamp"  : current_time,
                        "meta"       : {"direction": direction}
                    }
            prev_palm = palm

        elif gesture == "POINT_LEFT" and can_trigger:
            last_trigger_time = current_time
            event             = {
                "gesture"    : "POINT_LEFT",
                "hand_count" : len(all_hands),
                "confidence" : 0.0,
                "palm"       : {"x": palm[0], "y": palm[1]},
                "timestamp"  : current_time,
                "meta"       : {}
            }

        elif gesture == "POINT_RIGHT" and can_trigger:
            last_trigger_time = current_time
            event             = {
                "gesture"    : "POINT_RIGHT",
                "hand_count" : len(all_hands),
                "confidence" : 0.0,
                "palm"       : {"x": palm[0], "y": palm[1]},
                "timestamp"  : current_time,
                "meta"       : {}
            }

        elif gesture == "V_SIGN" and can_trigger and prev_gesture != "V_SIGN":
            photo_path        = f"{PHOTO_DIR}/gesture_photo_{int(current_time)}.jpg"
            last_trigger_time = current_time
            event             = {
                "gesture"    : "V_SIGN",
                "hand_count" : len(all_hands),
                "confidence" : 0.0,
                "palm"       : {"x": palm[0], "y": palm[1]},
                "timestamp"  : current_time,
                "meta"       : {"photo_path": photo_path}
            }

        elif gesture == "OPEN_HAND" and can_trigger and prev_gesture != "OPEN_HAND":
            last_trigger_time = current_time
            event             = {
                "gesture"    : "OPEN_HAND",
                "hand_count" : len(all_hands),
                "confidence" : 0.0,
                "palm"       : {"x": palm[0], "y": palm[1]},
                "timestamp"  : current_time,
                "meta"       : {"app": "app.zen_browser.zen"}
            }

        if gesture != "FIST":
            prev_palm = None

        prev_gesture = gesture

    else:
        prev_palm    = None
        prev_gesture = "NONE"

    return event, prev_palm, prev_gesture, last_trigger_time, last_move_time
