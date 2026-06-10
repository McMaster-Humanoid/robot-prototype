import data_transfer as dt
import argparse
import os
import json
from ultralytics import YOLO

#To run, type "python data_transfer_test.py -i <file_name>"
#I used path.os.join implementiation to make it more robust
#  but if it's too slow, hard coding the paths would be 10% faster  


#1. Test resize function, show size before and after, including file size
#2. Test predict image function based on file name 
#3. Test results of original vs compressed image of prediction

def calculate_reduction(original, result):
    percentage = ((original-result)/original)*100
    return percentage


def test_resize(input_list , input_dir, out_file_type, vlaX, vlaY, quality ):
    print(f"\n#####TESTING RESIZING FUNCTIONS (QUALITY = {quality})########\n")
    for _, image in enumerate(input_list):    
        print(f"File: {image}" )
        resize_path = dt.resize_for_vla(input_dir, image, out_file_type, vlaX, vlaY, quality)
        original_path = os.path.join(input_dir, image)

        h1, w1 = dt.find_dimensions(original_path)  
        h2, w2 = dt.find_dimensions(resize_path)  

        s1 = dt.find_file_details(original_path)
        s2 = dt.find_file_details(resize_path)
        
        print(f"Original Size: {h1}x{w1}")
        print(f"Reduced Size: {h2}x{w2}")
        print(f"Size: {s1} bytes --> {s2} bytes")
        reduction = calculate_reduction(s1,s2)
        print(f"Reduced file size by {round(reduction,2)}%\n")


def test_predict_compressed_image(model, input_list, out_file_type):
    print(f"\n############ TESTING PREDICTION ACCURACY ############\n")
    for _, image in enumerate(input_list):
        print(f"File: {image}")
        name, _ = os.path.splitext(image)
        dt.predict_image(model, "resized_images", out_file_type, name)
        metadata_path = os.path.join("metadata", f"{name}_metadata.json" )
        compressed_findings = dt.find_prediction(metadata_path)

        dt.predict_original_image(model, "images", image)
        original_path = os.path.join("metadata", f"original_{name}_metadata.json" )
        original_findings = dt.find_prediction(original_path)

        print(f"Expected: {name}")
        print(f"Original File Prediction: {original_findings}")
        print(f"Compressed File Prediction: {compressed_findings}\n")



        








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--input_dir", type = str, help = "Images folder", default = "images")
    parser.add_argument("-i", "--input_file", type = str, help = "Input image", nargs='*')
    parser.add_argument("-ft", "--out_file_type", type = str.lower, help = "Output file type", choices=["jpeg","webp", "jpg"], default = "jpeg")
    parser.add_argument("-x", "--vla_xinput", type = int, help = "VLA X Input Size", default = 224)
    parser.add_argument("-y", "--vla_yinput", type = int, help = "VLA Y Input Size", default = 224)
    parser.add_argument("-q", "--quality", type = int, help = "Compression quality from 0-100", default = 60)
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")

    if os.path.exists("./resized_images"):
        pass
    else: 
        os.makedirs("./resized_images", exist_ok=True)
        
    if os.path.exists("./metadata"):
        pass
    else: 
        os.makedirs("./metadata", exist_ok=True)



    test_resize(args.input_file, args.input_dir, args.out_file_type, args.vla_xinput, args.vla_yinput, args.quality)
    test_predict_compressed_image(model, args.input_file, args.out_file_type)







if __name__ == "__main__":
    main()


