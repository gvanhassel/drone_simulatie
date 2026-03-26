"""Simulatieklok voor framerate-onafhankelijke delta-time berekening."""

import time


class SimClock:
    """Beheert simulatietijd en levert consistente delta-time per frame."""

    def __init__(self, target_fps: int = 60) -> None:
        self.target_fps = target_fps
        self._target_dt: float = 1.0 / target_fps
        self._last_time: float = time.monotonic()
        self._elapsed: float = 0.0
        self._frame_count: int = 0
        self._fps_samples: list[float] = []

    def tick(self) -> float:
        """
        Geeft de delta-time (seconden) terug sinds de laatste aanroep.
        Begrensd op max. 3× target_dt om spiraaleffect bij lag te voorkomen.
        """
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        # Begrens dt bij grote pieken (bijv. bij de eerste frame of pauze)
        dt = min(dt, self._target_dt * 3)

        self._elapsed += dt
        self._frame_count += 1
        self._fps_samples.append(dt)
        if len(self._fps_samples) > 60:
            self._fps_samples.pop(0)

        return dt

    @property
    def elapsed(self) -> float:
        """Totale verstreken simulatietijd in seconden."""
        return self._elapsed

    @property
    def frame_count(self) -> int:
        """Aantal verwerkte frames."""
        return self._frame_count

    @property
    def fps(self) -> float:
        """Gemiddelde FPS over de laatste 60 frames."""
        if not self._fps_samples:
            return 0.0
        avg_dt = sum(self._fps_samples) / len(self._fps_samples)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    def reset(self) -> None:
        """Reset de klok (nuttig voor deterministisch testen)."""
        self._last_time = time.monotonic()
        self._elapsed = 0.0
        self._frame_count = 0
        self._fps_samples.clear()
