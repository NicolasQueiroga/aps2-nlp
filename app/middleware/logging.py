from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
import uuid
import time
from app.core.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request_start_time = time.time()

        logger.info(
            "Request received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
            },
        )

        try:
            response: Response = await call_next(request)
            process_time = time.time() - request_start_time
            logger.info(
                "Request processed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": f"{process_time:.4f} seconds",
                },
            )
            if response.status_code == 422:
                logger.error(
                    "Validation error occurred",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "url": str(request.url),
                    },
                )

            return response

        except Exception as e:
            logger.error(
                "Exception during request processing",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "exception": str(e),
                },
                exc_info=True,
            )
            return Response(
                content="Internal Server Error",
                status_code=500,
                media_type="application/json",
            )
