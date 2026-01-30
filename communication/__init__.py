"""
NAE Communication Module
========================
Provides secure communication channels for NAE agents.

Available:
- telegram_bot: NAE Secure Line for phone notifications
- casey_telegram_interface: Two-way communication with Casey
"""

from communication.telegram_bot import (
    NAETelegramBot,
    AlertPriority,
    get_telegram_bot,
    send_telegram_alert,
    send_trade_notification,
    send_error_notification
)

from communication.casey_telegram_interface import (
    CaseyTelegramInterface
)

__all__ = [
    'NAETelegramBot',
    'AlertPriority',
    'get_telegram_bot',
    'send_telegram_alert',
    'send_trade_notification',
    'send_error_notification',
    'CaseyTelegramInterface'
]

