"""User and subscription endpoints matching ElevenLabs API."""

import time

from fastapi import APIRouter

router = APIRouter(tags=["user"])


@router.get("/user/subscription")
async def get_subscription():
    """Mock subscription info for ElevenLabs client compatibility."""
    next_reset = int(time.time()) + 30 * 24 * 3600  # 30 days from now
    return {
        "tier": "creator",
        "character_count": 0,
        "character_limit": 100000,
        "max_character_limit_extension": 0,
        "can_extend_character_limit": False,
        "allowed_to_extend_character_limit": False,
        "next_character_count_reset_unix": next_reset,
        "voice_slots_used": 0,
        "professional_voice_slots_used": 0,
        "voice_limit": 100,
        "max_voice_add_edits": 100,
        "voice_add_edit_counter": 0,
        "professional_voice_limit": 0,
        "can_extend_voice_limit": False,
        "can_use_instant_voice_cloning": False,
        "can_use_professional_voice_cloning": False,
        "currency": "usd",
        "status": "active",
        "billing_period": "monthly_period",
        "character_refresh_period": "monthly_period",
        "next_invoice": {
            "amount_due_cents": 0,
            "subtotal_cents": 0,
            "tax_cents": 0,
            "discounts": [],
            "next_payment_attempt_unix": -1,
            "payment_intent_statusses": [],
        },
        "open_invoices": [],
        "has_open_invoices": False,
    }


@router.get("/user")
async def get_user():
    """Mock user info for ElevenLabs client compatibility."""
    return {
        "subscription": (await get_subscription()),
        "xi_api_key": "local-tts-server",
        "is_new_user": False,
        "can_use_delayed_payment_methods": False,
        "first_name": "Local",
    }
