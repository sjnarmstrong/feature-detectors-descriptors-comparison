import numpy as np
import cv2
from matplotlib import pyplot as plt

dataset_path = "../Datasets/Robots_ox_ac_uk/bark/"
img1 = cv2.imread(dataset_path + "img1.ppm")
img2 = cv2.imread(dataset_path + "img2.ppm")

# Initiate SIFT detector
sift = cv2.xfeatures2d_SIFT.create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# create BFMatcher object
bf = cv2.BFMatcher()
matches = bf.knnMatch(des2, des1, k=2)

# store all the good matches as per Lowe's ratio test.
good_matches = []
correct_match_count = 0
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)
match_count = len(good_matches)
good_matches = sorted(good_matches, key=lambda x: x.distance)
img_out = cv2.drawMatches(img1, kp2, img2, kp1, good[:40], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                          matchColor=(0, 0, 255))
plt.imshow(cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB))
plt.show()