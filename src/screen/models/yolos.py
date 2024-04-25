import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

COLORS = [
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]
]


class ObjectDetector:
    def __init__(self, model_path):
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(model_path)

    def detect(self, img_path, threshold=0.8):
        img = Image.open(img_path)
        inputs = self.image_processor(images=img, return_tensors='pt')
        outputs = self.model(**inputs)

        target_size = torch.tensor([img.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold, target_size=target_size)[0]

        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label = self.model.config.id2label[label.item()]
            if label in ['person', 'cell_phone', 'mask']:
                detected_objects.append({
                    "label": label,
                    "confidence": round(score.item(), 3),
                    "location": box
                })
        return detected_objects

    def draw_outputs(self, image, detected_objects):
        colors = [(255, 0, 0), (0, 255, 0)]  # Red for person, Green for cell_phone
        for obj, color in zip(detected_objects, colors):
            xmin, ymin, xmax, ymax = obj["location"]
            score = obj["confidence"]
            label = obj["label"]

            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(image, f'{label}: {score}', (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
