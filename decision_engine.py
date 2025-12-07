# decision_engine.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

# VIDEO: classes that should ALWAYS trigger alert if detected
VIDEO_EMERGENCY_LABELS = {"fire", "weapon", "accident", "thief", "fall"}

# AUDIO: based on your model.labels_: ['glass', 'gunshot', 'help', 'scream']
AUDIO_EMERGENCY_LABELS = {"glass", "gunshot", "help", "scream"}

VIDEO_MIN_CONF = 0.50     # you can tune
AUDIO_MIN_CONF = 0.60     # you can tune


@dataclass
class DecisionResult:
    should_alert: bool
    source_type: str        # "video" or "audio"
    label: str
    confidence: float
    reason: str
    payload: Dict[str, Any]  # extra info you can send to backend later


class DecisionEngine:
    """
    Very simple engine:
    - Video emergency -> always alert (if label in VIDEO_EMERGENCY_LABELS and conf >= VIDEO_MIN_CONF)
    - Audio emergency -> alert too (if label in AUDIO_EMERGENCY_LABELS and conf >= AUDIO_MIN_CONF)
    No waiting, no blocking, video does not depend on audio.
    """

    def handle_video(self, label: str, conf: float, extra: Dict[str, Any] | None = None) -> DecisionResult:
        extra = extra or {}
        if label in VIDEO_EMERGENCY_LABELS and conf >= VIDEO_MIN_CONF:
            reason = f"video emergency: {label} (conf={conf:.2f})"
            return DecisionResult(
                should_alert=True,
                source_type="video",
                label=label,
                confidence=conf,
                reason=reason,
                payload=extra,
            )
        else:
            reason = f"video non-emergency or low conf: {label} (conf={conf:.2f})"
            return DecisionResult(
                should_alert=False,
                source_type="video",
                label=label,
                confidence=conf,
                reason=reason,
                payload=extra,
            )

    def handle_audio(self, label: str, conf: float, extra: Dict[str, Any] | None = None) -> DecisionResult:
        extra = extra or {}
        if label in AUDIO_EMERGENCY_LABELS and conf >= AUDIO_MIN_CONF:
            reason = f"audio emergency: {label} (conf={conf:.2f})"
            return DecisionResult(
                should_alert=True,
                source_type="audio",
                label=label,
                confidence=conf,
                reason=reason,
                payload=extra,
            )
        else:
            reason = f"audio non-emergency or low conf: {label} (conf={conf:.2f})"
            return DecisionResult(
                should_alert=False,
                source_type="audio",
                label=label,
                confidence=conf,
                reason=reason,
                payload=extra,
            )


# Global singleton so you can simply: from decision_engine import engine
engine = DecisionEngine()
