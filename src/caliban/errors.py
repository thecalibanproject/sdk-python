"""Typed error hierarchy.

Every error raised by the SDK derives from :class:`CalibanError`. HTTP errors are
parsed from the OpenAPI ``Error`` shape::

    {"error": {"message": "...", "type": "policy_violation", "code": null}}

Hierarchy::

    CalibanError
    ├── APIConnectionError
    │   └── APITimeoutError
    └── APIError                      (message, type, code, request_id, body)
        ├── StreamError               (error event received mid-stream)
        ├── APIResponseValidationError (2xx body does not match the contract)
        └── APIStatusError            (+ status_code, response)
            ├── BadRequestError           400
            ├── AuthenticationError       401
            ├── PermissionDeniedError     403
            ├── PolicyViolationError      type == "policy_violation" (any status)
            ├── NotFoundError             404
            ├── ConflictError             409
            ├── UnprocessableEntityError  422
            ├── RateLimitError            429 or type == "rate_limited"
            └── InternalServerError       >= 500
                ├── UpstreamError         502 or type == "upstream_error"
                └── ServiceUnavailableError 503
"""

from __future__ import annotations

import json
from typing import Any

import httpx

__all__ = [
    "APIConnectionError",
    "APIError",
    "APIResponseValidationError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "CalibanError",
    "ConflictError",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "PolicyViolationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "StreamError",
    "UnprocessableEntityError",
    "UpstreamError",
]


class CalibanError(Exception):
    """Base class for every error raised by the Caliban SDK."""


class APIConnectionError(CalibanError):
    """The request never produced an HTTP response (DNS, TCP, TLS, reset...)."""

    def __init__(self, message: str = "Connection error.", *, request: httpx.Request | None = None):
        super().__init__(message)
        self.request = request


class APITimeoutError(APIConnectionError):
    """The request timed out."""

    def __init__(
        self, message: str = "Request timed out.", *, request: httpx.Request | None = None
    ):
        super().__init__(message, request=request)


class APIError(CalibanError):
    """An error reported by Caliban, carrying the OpenAPI ``Error`` fields."""

    def __init__(
        self,
        message: str,
        *,
        type: str | None = None,
        code: str | None = None,
        request_id: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.type = type
        self.code = code
        self.request_id = request_id
        self.body = body

    def __str__(self) -> str:
        parts = [self.message]
        if self.type:
            parts.append(f"type={self.type}")
        if self.code:
            parts.append(f"code={self.code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class StreamError(APIError):
    """An error payload was received after a stream had already started."""


class APIResponseValidationError(APIError):
    """A successful response whose body does not match the expected schema."""


class APIStatusError(APIError):
    """A non-2xx HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        response: httpx.Response,
        type: str | None = None,
        code: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(
            message,
            type=type,
            code=code,
            request_id=response.headers.get("x-caliban-request-id"),
            body=body,
        )
        self.response = response
        self.status_code = response.status_code

    def __str__(self) -> str:
        return f"{self.status_code}: {super().__str__()}"


class BadRequestError(APIStatusError):
    pass


class AuthenticationError(APIStatusError):
    pass


class PermissionDeniedError(APIStatusError):
    pass


class PolicyViolationError(APIStatusError):
    """Caliban refused the request on policy grounds (PII, trust tier, budget, ...)."""


class NotFoundError(APIStatusError):
    pass


class ConflictError(APIStatusError):
    pass


class UnprocessableEntityError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    pass


class InternalServerError(APIStatusError):
    pass


class UpstreamError(InternalServerError):
    """The upstream (BYOK) provider failed or returned garbage."""


class ServiceUnavailableError(InternalServerError):
    pass


_BY_TYPE: dict[str, type[APIStatusError]] = {
    "policy_violation": PolicyViolationError,
    "rate_limited": RateLimitError,
    "upstream_error": UpstreamError,
    "authentication_error": AuthenticationError,
}

_BY_STATUS: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    502: UpstreamError,
    503: ServiceUnavailableError,
}


def parse_error_body(body: Any) -> tuple[str | None, str | None, str | None]:
    """Extract ``(message, type, code)`` from an OpenAPI ``Error`` body, leniently."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            typ = err.get("type")
            code = err.get("code")
            return (
                msg if isinstance(msg, str) else None,
                typ if isinstance(typ, str) else None,
                None if code is None else str(code),
            )
        if isinstance(err, str):
            return err, None, None
        msg = body.get("message")
        if isinstance(msg, str):
            return msg, None, None
    return None, None, None


def make_status_error(response: httpx.Response, body_bytes: bytes | None = None) -> APIStatusError:
    """Build the most specific :class:`APIStatusError` for a non-2xx response."""
    raw = body_bytes if body_bytes is not None else response.content
    body: Any
    try:
        body = json.loads(raw) if raw else None
    except (ValueError, UnicodeDecodeError):
        body = raw.decode("utf-8", errors="replace") if raw else None

    message, err_type, code = parse_error_body(body)
    if message is None:
        reason = response.reason_phrase or "error"
        message = (
            body if isinstance(body, str) and body else f"HTTP {response.status_code} {reason}"
        )

    cls: type[APIStatusError] | None = _BY_TYPE.get(err_type or "")
    if cls is None:
        cls = _BY_STATUS.get(response.status_code)
    if cls is None:
        cls = InternalServerError if response.status_code >= 500 else APIStatusError
    return cls(message, response=response, type=err_type, code=code, body=body)
