"""ffmpeg trim helper. ML owns this. Backend calls trim_clip during POST."""


def trim_clip(video_path: str, start_s: float, duration_s: float, out_path: str) -> str:
    """Cut a clip of exactly duration_s, with a short fade in and out.

    Input is the wrist camera's mp4v file from robot.cameras (640x480, 30 fps).
    Re-encode to H.264 + yuv420p + faststart: phones will not play mp4v, and the
    fades need a re-encode anyway. Return out_path.
    """
    del video_path, start_s, duration_s, out_path
    raise NotImplementedError
