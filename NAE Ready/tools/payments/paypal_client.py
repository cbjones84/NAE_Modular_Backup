"""PayPal payout helper used by Shredder.

This implementation uses the official PayPal Checkout SDK and defaults to the
sandbox environment unless ``mode='live'`` is supplied in the credentials.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency branch
    from paypalcheckoutsdk.core import (
        PayPalHttpClient,
        LiveEnvironment,
        SandboxEnvironment,
    )
    from paypalcheckoutsdk.payouts import PayoutsPostRequest
except Exception:  # pragma: no cover - optional dependency branch
    PayPalHttpClient = None  # type: ignore
    LiveEnvironment = None  # type: ignore
    SandboxEnvironment = None  # type: ignore
    PayoutsPostRequest = None  # type: ignore

try:  # pragma: no cover - optional dependency branch
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass


class PaypalClientError(RuntimeError):
    """Raised when a PayPal request cannot be fulfilled."""


@dataclass
class PaypalCredentials:
    client_id: str
    client_secret: str
    mode: str = "sandbox"


class PaypalPayoutClient:
    """Wrapper around the PayPal Checkout SDK payouts endpoint."""

    def __init__(self, credentials: PaypalCredentials):
        if PayPalHttpClient is None or SandboxEnvironment is None:
            raise PaypalClientError(
                "paypalcheckoutsdk is not installed. "
                "Add 'paypalcheckoutsdk' to requirements."
            )

        if credentials.mode == "live":
            environment = LiveEnvironment(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
            )
        else:
            environment = SandboxEnvironment(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
            )

        self.mode = credentials.mode
        self.client = PayPalHttpClient(environment)

    def create_payout(
        self,
        *,
        amount: str,
        currency: str,
        recipient_email: str,
        note: Optional[str] = None,
        sender_batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if PayoutsPostRequest is None:  # pragma: no cover - defensive guard
            raise PaypalClientError("PayPal SDK payouts module unavailable.")

        batch_id = sender_batch_id or f"nae_batch_{uuid.uuid4().hex}"
        request = PayoutsPostRequest()
        request.request_body(
            {
                "sender_batch_header": {
                    "sender_batch_id": batch_id,
                    "email_subject": "NAE Profit Distribution",
                },
                "items": [
                    {
                        "recipient_type": "EMAIL",
                        "amount": {"value": amount, "currency": currency},
                        "receiver": recipient_email,
                        "note": note or "Shredder profit distribution",
                        "sender_item_id": f"nae_item_{uuid.uuid4().hex[:8]}",
                    }
                ],
            }
        )

        logger.debug(
            "Submitting PayPal payout request: batch_id=%s amount=%s %s to %s",
            batch_id,
            amount,
            currency,
            recipient_email,
        )

        try:
            response = self.client.execute(request)
            result = getattr(response, "result", None)
            batch_header = getattr(result, "batch_header", None)
            payout_batch_id = getattr(batch_header, "payout_batch_id", batch_id)
            logger.info(
                "PayPal payout created: batch_id=%s (mode=%s)",
                payout_batch_id,
                self.mode,
            )
            response_dict = result.__dict__ if result is not None else {}
            return {
                "status": "success",
                "batch_id": payout_batch_id,
                "response": response_dict,
            }
        except Exception as exc:  # pragma: no cover - network branch
            logger.error("PayPal payout failed: %s", exc)
            raise PaypalClientError(str(exc)) from exc

