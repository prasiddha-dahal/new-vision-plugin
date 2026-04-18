import grpc
import json
import sys
import time
from concurrent import futures
from typing import Generator, Iterator

sys.path.insert(0, "./gen/python")

import plugin_pb2
import plugin_pb2_grpc
from plugin_pb2 import (
    PluginStatus,
    PluginType,
    DataType,
    PluginError,
    InitializeRequest,
    InitializeResponse,
    GetMetadataRequest,
    GetMetadataResponse,
    HealthRequest,
    HealthResponse,
    HandleEventRequest,
    HandleEventResponse,
    StreamRequest,
    StreamResponse,
    ShutdownRequest,
    ShutdownResponse,
)
from grpc_reflection.v1alpha import reflection
from hand_gesture import detect_from_jpeg

PORT: int = 50051

class VisionPlugin(plugin_pb2_grpc.PluginServiceServicer):
    
    def __init__(self) -> None:
        self.plugin_id  : str | None = None
        self.name       : str | None = None
        self.version    : str | None = None
        self.description: str | None = None
        self.status     : int        = PluginStatus.Value("PLUGIN_STATUS_READY")

    def Initialize(self, request: InitializeRequest, context: grpc.ServicerContext) -> InitializeResponse:
        self.plugin_id   = request.plugin_id
        self.name        = request.name
        self.version     = request.version
        self.description = request.description
        print(f"[vision] initialized — id: {self.plugin_id} name: {self.name}")
        return InitializeResponse(success=True)

    def GetMetadata(self, request: GetMetadataRequest, context: grpc.ServicerContext) -> GetMetadataResponse:
        return GetMetadataResponse(
            plugin_id   = self.plugin_id,
            name        = self.name,
            version     = self.version,
            description = self.description,
            type        = PluginType.Value("PLUGIN_TYPE_VISION"),
            inputs      = [DataType.Value("DATA_TYPE_CAMERA")],
            outputs     = [DataType.Value("DATA_TYPE_JSON")]
        )

    def Health(
        self,
        request: HealthRequest,
        context: grpc.ServicerContext
    ) -> HealthResponse:
        return HealthResponse(
            status=self.status,
            message="ok"
        )

    def Stream(
        self,
        request_iterator: Iterator[StreamRequest],
        context: grpc.ServicerContext
    ) -> Generator[StreamResponse, None, None]:
        print("[vision] stream started")
        self.status = PluginStatus.Value("PLUGIN_STATUS_BUSY")

        last_trigger_time: float = 0.0
        last_move_time: float = 0.0
        prev_palm: tuple | None = None
        prev_gesture: str = "NONE"

        try:
            for request in request_iterator:
                if not context.is_active():
                    print("[vision] client disconnected")
                    break

                if request.data_type != DataType.Value("DATA_TYPE_CAMERA"):
                    continue

                current_time = time.time()

                event, prev_palm, prev_gesture, last_trigger_time, last_move_time = detect_from_jpeg(
                    jpeg_bytes=request.payload,
                    current_time=current_time,
                    prev_palm=prev_palm,
                    prev_gesture=prev_gesture,
                    last_trigger_time=last_trigger_time,
                    last_move_time=last_move_time,
                )

                if event is None:
                    continue

                print(f"[vision] detected: {event['gesture']}")

                yield StreamResponse(
                    data_type=DataType.Value("DATA_TYPE_JSON"),
                    payload=json.dumps(event).encode("utf-8")
                )

        except Exception as e:
            yield StreamResponse(
                error=PluginError(
                    code="STREAM_ERROR",
                    message=str(e)
                )
            )

        finally:
            self.status = PluginStatus.Value("PLUGIN_STATUS_READY")
            print("[vision] stream ended")

    def HandleEvent(
        self,
        request: HandleEventRequest,
        context: grpc.ServicerContext
    ) -> HandleEventResponse:
        event_type: str = request.event_type

        if event_type == "pause":
            return HandleEventResponse(success=True)

        if event_type == "resume":
            return HandleEventResponse(success=True)

        return HandleEventResponse(
            success=False,
            error=PluginError(
                code="UNKNOWN_EVENT",
                message=f"unknown event type: {event_type}"
            )
        )

    def Shutdown(
        self,
        request: ShutdownRequest,
        context: grpc.ServicerContext
    ) -> ShutdownResponse:
        print("[vision] shutting down")
        self.status = PluginStatus.Value("PLUGIN_STATUS_LOADING")
        return ShutdownResponse(success=True)


def serve() -> None:
    server: grpc.Server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    plugin_pb2_grpc.add_PluginServiceServicer_to_server(
        VisionPlugin(), server
    )

    SERVICE_NAMES = (
        plugin_pb2.DESCRIPTOR.services_by_name["PluginService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"[vision] server running on port {PORT}")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        print("[vision] stopped")


if __name__ == "__main__":
    serve()
