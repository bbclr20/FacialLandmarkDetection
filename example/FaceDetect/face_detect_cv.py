"""
This is an example from dlib and you can visit http://dlib.net/.

The detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
The objects can be accessed by simply iterating over the mmod_rectangles object
The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

It is also possible to pass a list of images to the detector.
    - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)

In this case it will return a mmod_rectangless object.
This object behaves just like a list of lists and can be iterated over.

You can get the mmod_human_face_detector.dat file from:
http://dlib.net/files/mmod_human_face_detector.dat.bz2
"""
import cv2
import dlib
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Face detection")
parser.add_argument('--image', type=str, default="../../data/image/lakers.jpg", help="Image with human face")
parser.add_argument('--dat', type=str, default="../../data/dat/mmod_human_face_detector.dat", help="Default dat file")
args = parser.parse_args()

dat_file = args.dat
image = args.image

#
# Dlib detect the face in RGB space
# OpenCV works in BGR space
#
img = cv2.imread(image) # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = dlib.load_rgb_image(image) # RGB

cnn_face_detector = dlib.cnn_face_detection_model_v1(dat_file)

face_rects = cnn_face_detector(img, 1)

print("Number of faces detected: {}".format(len(face_rects)))

for i, d in enumerate(face_rects):
  x1 = d.rect.left()
  y1 = d.rect.top()
  x2 = d.rect.right()
  y2 = d.rect.bottom()
  text = "{:2.2f}".format(d.confidence)

  cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4, cv2.LINE_AA)
  cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX,
              0.7, (255, 0, 0), 1, cv2.LINE_AA)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img)
# plt.axis("off")
# plt.show()