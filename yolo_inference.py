
from ultralytics import YOLO 

model = YOLO('models/last.pt')

result = model.track('input_videos/input_video.mp4',conf=0.2, save=True,save_dir="./")

# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)