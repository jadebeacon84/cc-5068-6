"""语音活动区间分析器 (voice-activity interval analyzer).

The package deliberately separates three layers:

* frame-level scoring  -- :mod:`vad_analyzer.features`,
  :mod:`vad_analyzer.detector`;
* interval merging      -- :mod:`vad_analyzer.segments`;
* evidence/session flow -- :mod:`vad_analyzer.stream`,
  :mod:`vad_analyzer.cuts`.

"voice activity" here means *speech-like acoustic activity estimated by a
local model/rules*; a loud sound is never named "speech" solely because it is
loud. See README for scope and limitations.
"""
from __future__ import annotations

from . import _env  # noqa: F401  (must run before numpy imports)

__version__ = "1.0.0"
