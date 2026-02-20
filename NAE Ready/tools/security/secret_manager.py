"""Secret management utilities for NAE.

This module provides a thin abstraction around the credential stores used by
the agents.  The default behaviour is:

1. Attempt to use AWS Secrets Manager when the ``boto3`` SDK and the
   ``NAE_AWS_SECRET_NAME`` metadata are available.
2. Fallback to a JSON-encoded environment variable for developer / CI usage.

The goal is to keep sensitive credentials (e.g. broker or payment processor
tokens) out of repository files and local configuration artefacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


class SecretRetrievalError(RuntimeError):
    """Raised when a secret cannot be located."""


def _attempt_import_boto3():
    try:
        import boto3  # type: ignore
        return boto3
    except Exception:
        return None


_BOTO3 = _attempt_import_boto3()


@dataclass
class SecretDescriptor:
    """Metadata describing where a secret is stored."""

    name: str
    region: Optional[str] = None
    env_var: Optional[str] = None


class SecretManager:
    """Retrieve secrets from AWS Secrets Manager or environment variables."""

    def __init__(self, descriptor: SecretDescriptor):
        self.descriptor = descriptor

    def get_secret(self) -> Dict[str, Any]:
        """Fetch and decode a JSON secret."""
        secret_payload: Optional[Dict[str, Any]] = None

        # Attempt AWS Secrets Manager when boto3 is available and region/name
        # metadata has been provided.
        if (
            _BOTO3 is not None
            and self.descriptor.name
            and self.descriptor.region
        ):
            session = _BOTO3.session.Session()
            client = session.client(
                service_name="secretsmanager", region_name=self.descriptor.region
            )
            try:
                response = client.get_secret_value(SecretId=self.descriptor.name)
                secret_string = response.get("SecretString")
                if secret_string:
                    secret_payload = json.loads(secret_string)
            except Exception as exc:  # pragma: no cover - network branch
                raise SecretRetrievalError(
                    f"Unable to fetch secret '{self.descriptor.name}' "
                    f"from AWS Secrets Manager ({exc})."
                ) from exc

        # Fallback to environment variable (JSON-encoded) if provided.
        if secret_payload is None and self.descriptor.env_var:
            env_value = os.getenv(self.descriptor.env_var)
            if env_value:
                try:
                    secret_payload = json.loads(env_value)
                except json.JSONDecodeError as exc:
                    raise SecretRetrievalError(
                        f"Environment variable '{self.descriptor.env_var}' "
                        f"does not contain valid JSON."
                    ) from exc

        if secret_payload is None:
            raise SecretRetrievalError(
                "No secret source available. Ensure AWS credentials are configured "
                "or supply the JSON payload via the designated environment variable."
            )

        return secret_payload


def load_paypal_sandbox_credentials() -> Dict[str, Any]:
    """Convenience helper for the sandbox PayPal credentials used by Shredder."""
    descriptor = SecretDescriptor(
        name=os.getenv("NAE_PAYPAL_SANDBOX_SECRET_NAME", ""),
        region=os.getenv("NAE_PAYPAL_SANDBOX_SECRET_REGION"),
        env_var=os.getenv("NAE_PAYPAL_SANDBOX_ENV_VAR", "NAE_PAYPAL_SANDBOX_CREDENTIALS"),
    )
    manager = SecretManager(descriptor)
    return manager.get_secret()



