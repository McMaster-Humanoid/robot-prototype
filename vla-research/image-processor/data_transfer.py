import cv2
import argparse
import os
import json


"""
Data Transfer - Jason
Resize the image to match the input size expected by the VLA vision encoder
Compress the image!! (e.g., JPEG/WebP) so it can be efficiently stored or transmitted
Test implementing object detection to send as metadata with YOLO

Typical VLA image size: 224x224
"""


def resize_for_vla(input_directory, 
                   input_file, 
                   output_file_type, 
                   resized_width, 
                   resized_height, 
                   compression_quality):
    
    name, _ = os.path.splitext(input_file)
    read_path = os.path.join(input_directory, input_file)
    img = cv2.imread(read_path)

    resized_image = cv2.resize(img, (resized_width, resized_height))


    if output_file_type in ["jpg", "jpeg"]:
        quality = [cv2.IMWRITE_JPEG_QUALITY, compression_quality]
    elif output_file_type == "webp":
        quality = [cv2.IMWRITE_WEBP_QUALITY, compression_quality]


    path = os.path.join("resized_images", f"{name}_resized.{output_file_type}")

    cv2.imwrite(path, resized_image, quality)
    return path


#Saves the results with at least 25% confidence into a json file
def predict_image(model, img_dir, file_type, name):
    
    results = model.predict(
        source = os.path.join(img_dir, f"{name}_resized.{file_type}"),
        conf=0.25,
        verbose=False
    )
    result = results[0] 

    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": confidence,
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    metadata = {
        "image": f"{name}_resized.{file_type}",
        "num_detections": len(detections),
        "detections": detections
    }
    output_file_path = os.path.join("metadata",f"{name}_metadata.json")
    with open(output_file_path, "w") as f:
        json.dump(metadata, f, indent=4)

#Saves the results with at least 25% confidence into a json file
def predict_original_image(model, img_dir, file_name):

    results = model.predict(
        source= os.path.join(img_dir, file_name),
        conf=0.25,
        verbose=False
    )
    result = results[0] 
    name, file_type = os.path.splitext(file_name)

    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": confidence,
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    metadata = {
        "image": f"{name}_resized{file_type}",
        "num_detections": len(detections),
        "detections": detections
    }

    output_file_path = os.path.join("metadata", f"original_{name}_metadata.json")
    with open(output_file_path, "w") as f:
        json.dump(metadata, f, indent=4)

#finds file size in bytes
def find_file_details(directory):
    size = os.path.getsize(directory)
    return size


#inputs: car.webp, images
def find_dimensions(directory):
    img = cv2.imread(directory)
    return img.shape[:2]

#takes a json file, then outputs the 
def find_prediction(directory):
    with open(directory, 'r') as f:
        data = json.load(f)

    objects = []

    for index, _ in enumerate(data["detections"]):
        objects.append(data["detections"][index]["class_name"])
    return objects


def main():
    pass

if __name__ == "__main__":
    main()  