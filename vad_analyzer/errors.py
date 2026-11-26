class VadError(Exception):
    """Expected, user-facing error. GUIs surface its message directly."""


class AudioReadError(VadError):
    pass


class PlaybackUnavailable(VadError):
    pass


class AnnotationError(VadError):
    pass
