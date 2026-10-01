"""Receiver-conditioned state transfer between existing agents."""

from rclc.diagnostics import check
from rclc.hf import Agent, HFReceiver, sender_from_hf
from rclc.latent import latent_mass, latent_mass_sync
from rclc.message import Message
from rclc.transport import Delivery, SenderState, transfer, transfer_sync

bind = Agent

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "Delivery",
    "HFReceiver",
    "Message",
    "SenderState",
    "__version__",
    "bind",
    "check",
    "latent_mass",
    "latent_mass_sync",
    "sender_from_hf",
    "transfer",
    "transfer_sync",
]
