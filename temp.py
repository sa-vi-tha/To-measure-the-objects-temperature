import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


cap = cv2.VideoCapture(0)


model = YOLO('yolov8n.pt') 

def map_temperature(raw_value, scale_factor=0.1, offset=0):
    """Convert raw simulated thermal values to temperature."""
    return raw_value * scale_factor + offset

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    
    thermal_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    normalized_frame = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)

   
    results = model(frame)

   
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy() 
        classes = result.boxes.cls.cpu().numpy()  
        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box[:4]) 
            label = model.names[int(cls)]  

         
            object_region = thermal_frame[y1:y2, x1:x2]
            avg_temp = np.mean(object_region)  
            temperature = map_temperature(avg_temp) 

          
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label}: {temperature:.2f}C",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 
            )
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                heatmap, f"{label}: {temperature:.2f}C",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2 
            )

   
    cv2.imshow("Detected Objects", frame)

   
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

   
    plt.imshow(heatmap_rgb)
    plt.axis('off')  
    plt.pause(0.001)  


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

