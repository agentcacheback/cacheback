"""Receiver-conditioned state transfer between existing agents."""

from rcc.diagnostics import check
from rcc.hf import Agent, HFReceiver, sender_from_hf
from rcc.latent import latent_mass, latent_mass_sync
from rcc.message import Message
from rcc.transport import Delivery, SenderState, transfer, transfer_sync

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
