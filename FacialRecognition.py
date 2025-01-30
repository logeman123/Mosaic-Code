from facenet_pytorch import MTCNN, extract_face
import torch
from PIL import Image, ImageDraw


#Determine if an nvidia GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# If required, create a face detection pipeline using MTCNN:
#mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
mtcnn = MTCNN(keep_all=True, device=device)

#image
img = Image.open('images/President_Barack_Obama1.jpg')

# Detect faces
boxes, probs, points = mtcnn.detect(img, landmarks=True)

img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)
    for p in point:
        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
    extract_face(img, box, save_path='detected_face_{}.png'.format(i))
img_draw.save('annotated_faces.png')