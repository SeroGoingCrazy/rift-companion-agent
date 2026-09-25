"""``ensure_user``: find or create a user by nickname (web login; no passwords in v1)."""

from __future__ import annotations

from sqlalchemy import select

from booking_mcp.db import User, write_session
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import EnsureUserInput, UserOut

NAME = "ensure_user"
DESCRIPTION = (
    "Find the user with this nickname, creating it on first use; returns the user id "
    "that the other tools expect. Used by the web login."
)


def ensure_user(service: BookingService, args: EnsureUserInput) -> UserOut:
    with write_session(service.factory) as session:
        user = session.scalars(select(User).where(User.nickname == args.nickname)).first()
        created = user is None
        if user is None:
            user = User(nickname=args.nickname)
            session.add(user)
            session.flush()
        return UserOut(user_id=user.id, nickname=user.nickname, created=created)
