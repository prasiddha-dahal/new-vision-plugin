import grpc
import sys
import time

sys.path.insert(0, "./gen/python")

import cv2
import plugin_pb2
import plugin_pb2_grpc
from plugin_pb2 import DataType

SERVER_ADDRESS = "localhost:50051"

def frame_generator():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            _, jpeg = cv2.imencode(".jpg", frame)
            jpeg_bytes = jpeg.tobytes()

            yield plugin_pb2.StreamRequest(
                data_type = DataType.Value("DATA_TYPE_CAMERA"),
                payload   = jpeg_bytes
            )

            time.sleep(0.033)  # ~30fps
    finally:
        cap.release()


def run():
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub    = plugin_pb2_grpc.PluginServiceStub(channel)

    stub.Initialize(plugin_pb2.InitializeRequest(plugin_id="test-client"))
    print(f"[test] connected to {SERVER_ADDRESS}")
    print("[test] streaming frames — show your hand...")

    responses = stub.Stream(frame_generator())

    for response in responses:
        if response.data_type == DataType.Value("DATA_TYPE_JSON"):
            import json
            event = json.loads(response.payload)
            print(f"[test] gesture: {event['gesture']} | palm: {event['palm']}")


if __name__ == "__main__":
    run()
