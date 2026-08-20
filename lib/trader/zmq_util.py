import logging
from typing import Optional, Self

import zmq
import zmq.asyncio
from zmq import LINGER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZmqPubSub:
    """
    Simple ZMQ PubSub interface with single Context instance
    that can be used for both async and blocking sockets
    """

    instance: Self = None

    def __init__(self):
        self._ctx = zmq.Context.instance()
        self._async_ctx = zmq.asyncio.Context.shadow(self._ctx.underlying)

    @classmethod
    def singleton(cls) -> Self:
        if not ZmqPubSub.instance:
            ZmqPubSub.instance = cls()
        return ZmqPubSub.instance

    def __str__(self) -> str:
        return "ZMQ PubSub Singleton"

    def get_context(self, async_ctx: bool) -> zmq.Context:
        return self._async_ctx if async_ctx else self._ctx

    def clean_shutdown(self) -> None:
        try:
            self._ctx.term()
            self._async_ctx.term()
        except Exception:
            pass


def add_publisher(
        connect_url: str,
        async_ctx: bool = True) -> zmq.Socket:
    """
    Returns publisher socket connected to target address
    async_ctx used to determine whether async or blocking socket
    """
    socket = ZmqPubSub.singleton().get_context(async_ctx=async_ctx).socket(zmq.PUB)
    socket.setsockopt(LINGER, 0)
    socket.connect(connect_url)
    return socket


def add_pusher(
        connect_url: str) -> zmq.Socket:
    """
    Returns publisher socket connected to target address
    async_ctx used to determine whether async or blocking socket
    """
    socket = ZmqPubSub.singleton().get_context(async_ctx=False).socket(zmq.PUSH)
    socket.connect(connect_url)
    return socket


def add_subscriber(
        connect_url: str,
        msg_filter: Optional[bytes] = b'',
        async_ctx: bool = True) -> zmq.Socket:
    """
    Returns subscriber socket connected to target address
    async_ctx used to determine whether async or blocking socket
    """
    socket = ZmqPubSub.singleton().get_context(async_ctx=async_ctx).socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, msg_filter)
    socket.connect(connect_url)
    return socket
