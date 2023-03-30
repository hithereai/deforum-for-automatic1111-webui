from decimal import Decimal, getcontext

def time_to_srt_format(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{int(milliseconds * 1000):03}"

def init_srt_file(filename, fps, precision=20):
    with open(filename, "w") as f:
        pass
    getcontext().prec = precision
    frame_duration = Decimal(1) / Decimal(fps)
    return frame_duration

def write_frame_subtitle(filename, frame_number, frame_duration, text):
    frame_start_time = Decimal(frame_number) * frame_duration
    frame_end_time = (Decimal(frame_number) + Decimal(1)) * frame_duration

    with open(filename, "a") as f:
        f.write(f"{frame_number + 1}\n")
        f.write(f"{time_to_srt_format(frame_start_time)} --> {time_to_srt_format(frame_end_time)}\n")
        f.write(f"{text}\n\n")