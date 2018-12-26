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
import dlib
import argparse

parser = argparse.ArgumentParser(description="Face detection")
parser.add_argument('--image', type=str, default="../../data/image/lakers.jpg", help="Image with human face")
parser.add_argument('--dat', type=str, default="../../data/dat/mmod_human_face_detector.dat", help="Default dat file")
args = parser.parse_args()

dat_file = args.dat
image = args.image

cnn_face_detector = dlib.cnn_face_detection_model_v1(dat_file)
win = dlib.image_window()

print("Processing file: {}".format(image))
img = dlib.load_rgb_image(image)

# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
face_rects = cnn_face_detector(img, 1)

print("Number of faces detected: {}".format(len(face_rects)))

for i, d in enumerate(face_rects):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

rects = dlib.rectangles()
rects.extend([d.rect for d in face_rects])

win.clear_overlay()
win.set_image(img)
win.add_overlay(rects)
dlib.hit_enter_to_continue()
