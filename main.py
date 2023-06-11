import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Sadece kullanmak istediğiniz sınıfların indekslerini belirleyin
selected_classes = ['helmet', 'no-helmet']
selected_class_ids = [classes.index(cls) for cls in selected_classes]


output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(selected_classes), 3))

# Loading image
image = cv2.imread("images/img8.jpg")

font = cv2.FONT_HERSHEY_PLAIN
height, width, _ = image.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if class_id in selected_class_ids and confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[selected_class_ids.index(class_ids[i])]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Pencereyi yeniden boyutlandırma modunda açar
cv2.resizeWindow("Image", 800, 600)  # Pencereyi 800x600 boyutunda ayarlar

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
