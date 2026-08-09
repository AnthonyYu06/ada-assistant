"""Wake-word detection using openWakeWord ("hey ada").

The detector consumes the 1280-sample (80 ms @ 16 kHz) int16 frames that
MicStream produces and reports True once per activation, with a cooldown
so one long utterance of the phrase can't trigger repeatedly.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

log = logging.getLogger("ada.audio.wakeword")

# Trailing "_v0.1"-style version suffix on pretrained model names.
_VERSION_SUFFIX = re.compile(r"_v[\d.]+$")


class WakeWordDetector:
    """openWakeWord wrapper with thresholding and a detection cooldown."""

    def __init__(
        self,
        model_name: str = "hey_mycroft_v0.1",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._base_name = _VERSION_SUFFIX.sub("", model_name)
        self._last_score = 0.0
        self._last_detection = float("-inf")

        # Accept the name with or without the version suffix.
        candidates = [model_name]
        alternate = (
            self._base_name
            if self._base_name != model_name
            else f"{model_name}_v0.1"
        )
        if alternate not in candidates:
            candidates.append(alternate)

        try:
            self._model = self._construct(candidates)
        except Exception as first_exc:  # noqa: BLE001 - missing model files
            log.info(
                "Wake-word model %r not loadable (%s); downloading pretrained "
                "openWakeWord models",
                model_name,
                first_exc,
            )
            self.ensure_models(model_name)
            try:
                self._model = self._construct(candidates)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not load wake-word model {model_name!r} even after "
                    f"downloading the pretrained models: {exc}"
                ) from exc

    @staticmethod
    def _construct(candidates: list[str]):  # noqa: ANN205 - openwakeword Model
        from openwakeword.model import Model

        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                model = Model(
                    wakeword_models=[candidate], inference_framework="onnx"
                )
            except Exception as exc:  # noqa: BLE001 - try the next name
                last_exc = exc
                continue
            log.debug("Loaded wake-word model %r", candidate)
            return model
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def ensure_models(model_name: str) -> None:
        """Download openWakeWord's pretrained model files (first run only)."""
        try:
            import openwakeword.utils

            openwakeword.utils.download_models()
        except Exception as exc:  # noqa: BLE001 - usually a network problem
            raise RuntimeError(
                f"Could not download the openWakeWord model files needed for "
                f"wake-word detection ({model_name!r}). This one-time download "
                f"requires an internet connection — check your network and try "
                f"again. Underlying error: {exc}"
            ) from exc

    @property
    def last_score(self) -> float:
        """Most recent wake-word score (0..1) — for doctor/debug displays."""
        return self._last_score

    def process(self, frame: np.ndarray) -> bool:
        """Feed one 1-D int16 frame (1280 samples). True on a new detection."""
        scores = self._model.predict(frame)
        relevant = [
            float(v) for k, v in scores.items() if self._base_name in str(k)
        ]
        if not relevant:
            relevant = [float(v) for v in scores.values()]
        score = max(relevant, default=0.0)
        self._last_score = score

        if score < self.threshold:
            return False
        now = time.monotonic()
        if now - self._last_detection < self.cooldown_s:
            return False
        self._last_detection = now
        if hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception as exc:  # noqa: BLE001 - reset is best-effort
                log.debug("Wake-word model reset failed: %s", exc)
        log.info("Wake word detected (score=%.2f)", score)
        return True
