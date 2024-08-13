from pathlib import Path
import cv2
import sys
import multiprocessing


def overlay_print(s: str) -> None:
    sys.stdout.write(f'\r{s}')      # \r means go back to the beginning of the line
    sys.stdout.flush()


def progress_bar(progress: float, show_str: str = '', bar_length: int = 40,
                 finished_chr: str = '=', unfinished_chr: str = '-',
                 left_border_chr: str = '[', right_border_chr: str = ']',
                 progress_precision: int = 2) -> None:
    if progress > 1:
        progress = 1
    filled_length = int(progress * bar_length)
    bar = finished_chr * filled_length + unfinished_chr * (bar_length - filled_length)
    show_str = show_str if show_str else f'{progress * 100:.{progress_precision}f}%'
    overlay_print(f'{left_border_chr}{bar}{right_border_chr} {show_str}')


class VideoToFrames:
    def __init__(self, video_path: str | Path, save_dir: str | Path, frame_interval: int = 1,
                 show_progress: bool = True, workers: int = 5) -> None:
        self.video_path = video_path
        self.save_dir = Path(save_dir) / Path(video_path).stem
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_interval = frame_interval
        self.show_progress = show_progress
        self._progress = multiprocessing.Value('i', 0)
        self.workers = workers

        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_digits = len(str(self.total_frames))  # 计算帧数的填充位数

        indices = self._split_number_into_parts(self.total_frames, self.workers)
        for i in range(self.workers):
            multiprocessing.Process(target=self._save_process, args=(indices[i], indices[i + 1])).start()

    @staticmethod
    def _split_number_into_parts(number: int, parts: int) -> list[int]:
        if parts <= 0 or number <= 0 or not isinstance(parts, int) or not isinstance(number, int):
            raise ValueError("Number and parts should be int and greater than 0.")
        part_size = number // parts
        return [i * part_size for i in range(parts)] + [number]

    def _save_process(self, start_frame: int, stop_frames: int) -> None:
        cap = cv2.VideoCapture(self.video_path)
        while start_frame < stop_frames:
            ret, frame = cap.read()
            if start_frame % self.frame_interval == 0:
                frame_filename = self.save_dir / f"frame_{start_frame:0{self.num_digits}d}.png"
                cv2.imwrite(str(frame_filename), frame)

            start_frame += 1
            with self._progress.get_lock():
                self._progress.value += 1
                if self.show_progress:
                    progress_bar(self._progress.value / self.total_frames,
                                 show_str=f'{self._progress.value}/{self.total_frames}')
        cap.release()
