# Copied from the team's AI server repo (services/roport-ai-server omx_udp); keep the code identical there and here.
"""Policy backed by the OMX panel's LeRobot runner on the GPU PC (omx_tools/omx_infer.py).

The panel already owns model loading, pre/post-processing and the observation format; this adapter
only reuses it so UDP replies match its HTTP /load and /infer replies. Imported lazily: it needs the
panel's LeRobot/PyTorch environment, which the AI server's own tests never install.
"""

import sys
from pathlib import Path

from .server import PolicyError


class PanelPolicy:
    def __init__(self, tools_dir: Path):
        tools = Path(tools_dir).resolve()
        if not (tools / "omx_infer.py").is_file():
            raise ValueError(f"omx_infer.py not found in {tools}")
        sys.path.insert(0, str(tools))
        import omx_infer

        self._infer = omx_infer
        self._cache = omx_infer.ModelCache()

    def _runner(self, model: str, overrides: dict):
        if not self._infer.allowed_model(model):
            raise PolicyError("MODEL_NOT_ALLOWED", "model is not in the checkpoints")
        return self._cache.get(model, overrides)

    def model_info(self, model: str) -> dict:
        """What the host PC's panel needs before an evaluation, so it never reads the GPU PC over SSH:
        the model's config files (text) and its home pose (from the training data; None if it cannot be made)."""
        root = Path(model)
        files = {}
        for name in ("config.json", "train_config.json"):
            try:
                files[name] = (root / name).read_text()
            except OSError:
                pass
        info = {"files": files, "home": None}
        try:
            import omx_home

            info["home"] = omx_home.model_home(model)
        except Exception as exc:  # noqa: BLE001 - e.g. the training data is gone; the panel decides
            info["home_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return info

    def load(self, model: str, overrides: dict) -> dict:
        if (overrides or {}).get("_info_only"):
            # LOAD with overrides {"_info_only": true}: the files and home pose only, nothing loaded on the GPU
            if not self._infer.allowed_model(model):
                raise PolicyError("MODEL_NOT_ALLOWED", "model is not in the checkpoints")
            return {"ok": True, "info_only": True, **self.model_info(model)}
        runner, load_ms = self._runner(model, overrides)
        diffusion = runner.cfg.type == "diffusion"
        return {
            "ok": True,
            "load_ms": round(load_ms, 1),
            "type": runner.cfg.type,
            "image_size": runner.image_size(),
            "n_obs": int(getattr(runner.cfg, "n_obs_steps", 1) or 1)
            if diffusion
            else 1,
            **self.model_info(model),
        }

    def infer(self, header: dict, data: bytes) -> dict:
        runner, load_ms = self._runner(
            header.get("model", ""), header.get("overrides") or {}
        )
        hist = self._infer.unpack_observations(header["obs"], data)
        rtc = header.get("rtc") or {}
        processed, original = runner.infer(
            hist,
            header.get("task"),
            header.get("robot_type"),
            int(header.get("chunk") or 50),
            rtc.get("delay", 0),
            rtc.get("left"),
        )
        return {
            "actions": processed.tolist(),
            "original": original.tolist(),
            "load_ms": round(load_ms, 1),
        }

    def status(self) -> dict:
        return {"loaded": [key[0] for key in self._cache.items]}
