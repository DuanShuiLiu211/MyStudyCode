class YoloDetectorOut:
    __slots__ = ["class_index", "x1", "y1", "x2", "y2", "confidence", "class_name"]
    class_index: int = 0
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str


class ResnetClassifyOut:
    __slots__ = ["class_index", "confidence", "class_name"]
    class_index: int
    confidence: float
    class_name: str
