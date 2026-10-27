from .base import AlignmentLoss, get_alignment_loss, register
from .linear_mse import LinearMSEAlignmentLoss
from .cka import CKAAlignmentLoss
from .rsa import RSAAlignmentLoss

__all__ = [
    "AlignmentLoss",
    "get_alignment_loss",
    "register",
    "LinearMSEAlignmentLoss",
    "CKAAlignmentLoss",
    "RSAAlignmentLoss",
]
from .infonce import InfoNCEAlignmentLoss  # noqa: E402

__all__.append("InfoNCEAlignmentLoss")
