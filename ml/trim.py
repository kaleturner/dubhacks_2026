"""ffmpeg trim helper. ML owns this. Backend calls trim_clip during POST."""


def trim_clip(video_path: str, start_s: float, duration_s: float, out_path: str) -> str:
    """Stream-copy a clip of exactly duration_s, with a short fade in and out.

    Return out_path. Normalize webm/mov to mp4 when the input is not already mp4.
    """
    del video_path, start_s, duration_s, out_path
    raise NotImplementedError
