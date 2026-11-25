# Copied from the team's AI server repo (services/roport-ai-server/src/omx_udp, task/omx-cooperative c2baaa0); keep the code identical there and here.
"""Refusals shared by the server and the policies.

Kept apart from server.py: `python -m omx_udp.server` runs that file as __main__ while a policy
imports omx_udp.server again, which would make two PolicyError classes, and the server would
report a policy's refusal (e.g. MODEL_NOT_ALLOWED) as INFER_FAILED.
"""


class PolicyError(Exception):
    """A refusal with a stable code, e.g. MODEL_NOT_ALLOWED or NOT_LOADED."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code
