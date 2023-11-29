import cv2
from torch.utils.data import Dataset


class VideoReader(Dataset):
    def __init__(self, path, frame_rate=30):
        desired_fps = frame_rate
        cap = cv2.VideoCapture(path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / desired_fps)
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_list.append(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            for _ in range(frame_interval - 1):
                cap.read()
        cap.release()
        self.video = frame_list

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        frame = self.video[idx]
        return frame


class VideoReaderIter:
    def __init__(self, path, frame_rate=30):
        self.path = path
        self.frame_rate = frame_rate

    def __iter__(self):
        desired_fps = self.frame_rate
        cap = cv2.VideoCapture(self.path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / desired_fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            for _ in range(frame_interval - 1):
                cap.read()
        cap.release()
