
from roboflow import Roboflow
rf = Roboflow(api_key="a4OUT2gQsGbk0OzKs7j0")
project = rf.workspace("bin-ulwnh").project("detect-beer")
version = project.version(1)
dataset = version.download("yolov5")

!yolo task=detect mode=train model=yolov5s.pt data={dataset.location}/data.yaml epochs=100 imgsz=640