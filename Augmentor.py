import Augmentor

# Define the source and output directories
source_directory = "C:/Users/kashinath konade/Desktop/Bird Detection and Tracking/1"
output_directory = "C:/Users/kashinath konade/Desktop/Bird Detection and Tracking/2"

# Create a pipeline
p = Augmentor.Pipeline(source_directory, output_directory)

# Define augmentation operations
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)

# Set the number of augmented images you want to generate
num_samples = 20

p.sample(num_samples)



from ultralytics import YOLO
model = YOLO("yolov8m.pt")

r = model.predict(r"C:\Users\kashinath konade\Downloads\New folder\train\images\Monty-wants-to-play_jpg.rf.3f15d35da895ba1f7f8002100151e34f.jpg")

result = r[0]

print(len(result.boxes))

for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()
    print("Object type: ",  label)
    print("Coordinates: ", cords)
    print("Probability: ", prob)
    

from PIL import Image
Image.fromarray(result.plot()[:,:,::-1])




























