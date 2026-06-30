import asyncio

from slowapi.errors import RateLimitExceeded

import app as app_module


class _FakeLimit:
    error_message = None
    limit = "10 per 1 minute"


def test_rate_limit_handler_returns_429():
    exc = RateLimitExceeded(_FakeLimit())
    response = asyncio.run(app_module.rate_limit_handler(None, exc))
    assert response.status_code == 429
    assert response.body == (
        b'{"detail":"You\'re sending requests too quickly. '
        b'Please wait a moment and try again."}'
    )
