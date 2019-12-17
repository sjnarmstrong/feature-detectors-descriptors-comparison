import cv2
import numpy as np
from plotTools import Plotter
import json


def get_match_rate(kp1, kp2, des1, des2, kp_1_np, T_1_2, img1, img2, matcher_alg, thresh_sq, match_thresh=0.75,
                   approx_points_to_draw=80):
    kp_2_np = np.array(list(map(lambda x: x.pt + (1.0,), kp2)))

    kp_t_1_2 = np.dot(T_1_2, kp_1_np.T)
    # seems they occasionally encode the light levels into this matrix too? Docs unclear
    kp_t_1_2 = (kp_t_1_2 / kp_t_1_2[2]).T
    kp_t_1_2_test = np.dot(T_1_2, kp_1_np.T).T

    matches = matcher_alg.knnMatch(des1, des2, k=2)

    correct_matches = []
    incorrect_matches = []
    for match_1, match_2 in matches:
        if match_1.distance < match_thresh * match_2.distance:
            if (np.sum(np.square(kp_t_1_2[match_1.queryIdx, :2] - kp_2_np[match_1.trainIdx, :2])) < thresh_sq or
                    np.sum(np.square(kp_t_1_2_test[match_1.queryIdx, :2] - kp_2_np[match_1.trainIdx, :2])) < thresh_sq):
                correct_matches.append(match_1)
            else:
                incorrect_matches.append(match_1)
    correct_count = len(correct_matches)
    incorrect_match_count = len(incorrect_matches)
    found_match_count = correct_count + incorrect_match_count
    if found_match_count > 0:
        correct_matches = sorted(correct_matches, key=lambda x: x.distance)
        incorrect_matches = sorted(incorrect_matches, key=lambda x: x.distance)
        img_out = cv2.drawMatches(img1, kp1, img2, kp2,
                                  correct_matches[
                                  :int(0.5 + approx_points_to_draw * correct_count / found_match_count)],
                                  None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                  matchColor=(0, 255, 0))
        img_out = cv2.drawMatches(img1, kp1, img2, kp2,
                                  incorrect_matches[
                                  :int(0.5 + approx_points_to_draw * incorrect_match_count / found_match_count)],
                                  img_out,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG +
                                        cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                                  matchColor=(0, 0, 255))
    else:
        img_out = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2]),
                           dtype=img1.dtype)
        img_out[:img1.shape[0], :img1.shape[1], :img1.shape[2]] = img1[:, :, :]
        img_out[:img2.shape[0], img1.shape[1]:, :img2.shape[2]] = img2[:, :, :]

    return int(correct_count), found_match_count, img_out


class DatasetInfo:
    def __init__(self, dataset_path):
        self.img1 = cv2.imread(dataset_path + "img1.ppm")
        self.img2 = cv2.imread(dataset_path + "img2.ppm")
        self.img3 = cv2.imread(dataset_path + "img3.ppm")
        self.img4 = cv2.imread(dataset_path + "img4.ppm")
        self.img5 = cv2.imread(dataset_path + "img5.ppm")
        self.img6 = cv2.imread(dataset_path + "img6.ppm")
        if self.img1 is None:
            self.img1 = cv2.imread(dataset_path + "img1.pgm")
        if self.img2 is None:
            self.img2 = cv2.imread(dataset_path + "img2.pgm")
        if self.img3 is None:
            self.img3 = cv2.imread(dataset_path + "img3.pgm")
        if self.img4 is None:
            self.img4 = cv2.imread(dataset_path + "img4.pgm")
        if self.img5 is None:
            self.img5 = cv2.imread(dataset_path + "img5.pgm")
        if self.img6 is None:
            self.img6 = cv2.imread(dataset_path + "img6.pgm")

        if (self.img1 is None or self.img2 is None or self.img3 is None
                or self.img4 is None or self.img5 is None or self.img6 is None):
            print("Images not found in: " + dataset_path)
            print("Replacing images with trivial images, please ensure that the relevant dataset is extracted.")
            self.img1 = np.zeros((50, 50, 3), dtype=np.uint8)
            self.img2 = np.zeros((50, 50, 3), dtype=np.uint8)
            self.img3 = np.zeros((50, 50, 3), dtype=np.uint8)
            self.img4 = np.zeros((50, 50, 3), dtype=np.uint8)
            self.img5 = np.zeros((50, 50, 3), dtype=np.uint8)
            self.img6 = np.zeros((50, 50, 3), dtype=np.uint8)
        self.t_1_2 = np.loadtxt(dataset_path + "H1to2p")
        self.t_1_3 = np.loadtxt(dataset_path + "H1to3p")
        self.t_1_4 = np.loadtxt(dataset_path + "H1to4p")
        self.t_1_5 = np.loadtxt(dataset_path + "H1to5p")
        self.t_1_6 = np.loadtxt(dataset_path + "H1to6p")


def detect_key_points(f, img, max_iterations=20, number_of_points_min=350, number_of_points_max=500):
    non_max = hasattr(f, "setNonmaxSuppression")
    if non_max:
        f.setNonmaxSuppression(True)
    kp = f.detect(img)
    if not hasattr(f, "getThreshold") or not hasattr(f, "setThreshold"):
        return kp, non_max, None
    current_step_positive_dir = None
    current_step_size = f.getThreshold() / 10
    is_int_thresh = type(f.getThreshold()) == int
    for i in range(max_iterations):
        if len(kp) < number_of_points_min:
            if current_step_positive_dir:
                current_step_size = current_step_size / 10
            new_thresh = f.getThreshold() - current_step_size
            if is_int_thresh:
                new_thresh = int(new_thresh)
                if f.getThreshold() == new_thresh:
                    break
            f.setThreshold(new_thresh)
            current_step_positive_dir = False
        elif len(kp) > number_of_points_max:
            if current_step_positive_dir == False:
                current_step_size = current_step_size / 10
            new_thresh = f.getThreshold() + current_step_size
            if is_int_thresh:
                new_thresh = int(new_thresh)
                if f.getThreshold() == new_thresh:
                    break
            f.setThreshold(new_thresh)
            current_step_positive_dir = True
        else:
            break
        kp = f.detect(img)
    return kp, non_max, f.getThreshold()


def evaluate_detector_on_dataset(dataset: DatasetInfo, feature_det, feature_des, thresh_sq=25):
    kp1, non_max_1, thresh_1 = detect_key_points(feature_det, dataset.img1,
                                                 number_of_points_min=1500, number_of_points_max=3000)
    kp2, non_max_2, thresh_2 = detect_key_points(feature_det, dataset.img2,
                                                 number_of_points_min=3000, number_of_points_max=8000)
    kp3, non_max_3, thresh_3 = detect_key_points(feature_det, dataset.img3,
                                                 number_of_points_min=3000, number_of_points_max=8000)
    kp4, non_max_4, thresh_4 = detect_key_points(feature_det, dataset.img4,
                                                 number_of_points_min=3000, number_of_points_max=8000)
    kp5, non_max_5, thresh_5 = detect_key_points(feature_det, dataset.img5,
                                                 number_of_points_min=3000, number_of_points_max=8000)
    kp6, non_max_6, thresh_6 = detect_key_points(feature_det, dataset.img6,
                                                 number_of_points_min=3000, number_of_points_max=8000)

    kp1, desc1 = feature_des.compute(dataset.img1, kp1)
    kp2, desc2 = feature_des.compute(dataset.img1, kp2)
    kp3, desc3 = feature_des.compute(dataset.img1, kp3)
    kp4, desc4 = feature_des.compute(dataset.img1, kp4)
    kp5, desc5 = feature_des.compute(dataset.img1, kp5)
    kp6, desc6 = feature_des.compute(dataset.img1, kp6)

    kp_1_np = np.array(list(map(lambda x: x.pt + (1.0,), kp1)))

    matcher = cv2.BFMatcher(normType=feature_des.defaultNorm())

    correct_match_count_1_2, match_count_1_2, img_out_1_2 = \
        get_match_rate(kp1, kp2, desc1, desc2, kp_1_np, dataset.t_1_2, dataset.img1, dataset.img2, matcher, thresh_sq)
    correct_match_count_1_3, match_count_1_3, img_out_1_3 = \
        get_match_rate(kp1, kp3, desc1, desc3, kp_1_np, dataset.t_1_3, dataset.img1, dataset.img3, matcher, thresh_sq)
    correct_match_count_1_4, match_count_1_4, img_out_1_4 = \
        get_match_rate(kp1, kp4, desc1, desc4, kp_1_np, dataset.t_1_4, dataset.img1, dataset.img4, matcher, thresh_sq)
    correct_match_count_1_5, match_count_1_5, img_out_1_5 = \
        get_match_rate(kp1, kp5, desc1, desc5, kp_1_np, dataset.t_1_5, dataset.img1, dataset.img5, matcher, thresh_sq)
    correct_match_count_1_6, match_count_1_6, img_out_1_6 = \
        get_match_rate(kp1, kp6, desc1, desc6, kp_1_np, dataset.t_1_6, dataset.img1, dataset.img6, matcher, thresh_sq)

    return ([correct_match_count_1_2, correct_match_count_1_3, correct_match_count_1_4, correct_match_count_1_5,
             correct_match_count_1_6],
            [match_count_1_2, match_count_1_3, match_count_1_4, match_count_1_5, match_count_1_6],
            [img_out_1_2, img_out_1_3, img_out_1_4, img_out_1_5, img_out_1_6],
            [non_max_1, non_max_2, non_max_3, non_max_4, non_max_5, non_max_6],
            [thresh_1, thresh_2, thresh_3, thresh_4, thresh_5, thresh_6],
            [len(kp1), len(kp2), len(kp3), len(kp4), len(kp5), len(kp6)])


datasets = {"bark": "../Datasets/Robots_ox_ac_uk/bark/",
            "bikes": "../Datasets/Robots_ox_ac_uk/bikes/",
            "boat": "../Datasets/Robots_ox_ac_uk/boat/",
            "graf": "../Datasets/Robots_ox_ac_uk/graf/",
            "leuven": "../Datasets/Robots_ox_ac_uk/leuven/",
            "trees": "../Datasets/Robots_ox_ac_uk/trees/",
            "ubc": "../Datasets/Robots_ox_ac_uk/ubc/",
            "wall": "../Datasets/Robots_ox_ac_uk/wall/"}

feature_detectors = \
    {
        "AGAST": {
           "bark": cv2.AgastFeatureDetector.create(threshold=40, nonmaxSuppression=False),
           "bikes": cv2.AgastFeatureDetector.create(threshold=30, nonmaxSuppression=False),
           "boat": cv2.AgastFeatureDetector.create(threshold=60, nonmaxSuppression=False),
           "graf": cv2.AgastFeatureDetector.create(threshold=40, nonmaxSuppression=False),
           "leuven": cv2.AgastFeatureDetector.create(threshold=40, nonmaxSuppression=False),
           "trees": cv2.AgastFeatureDetector.create(threshold=60, nonmaxSuppression=False),
           "ubc": cv2.AgastFeatureDetector.create(threshold=50, nonmaxSuppression=False),
           "wall": cv2.AgastFeatureDetector.create(threshold=50, nonmaxSuppression=False)
        },
        "AKAZE": {
           "bark": cv2.AKAZE_create(),
           "bikes": cv2.AKAZE_create(),
           "boat": cv2.AKAZE_create(),
           "graf": cv2.AKAZE_create(),
           "leuven": cv2.AKAZE_create(),
           "trees": cv2.AKAZE_create(),
           "ubc": cv2.AKAZE_create(),
           "wall": cv2.AKAZE_create()
        },
        "BRISK": {
           "bark": cv2.BRISK.create(),
           "bikes": cv2.BRISK.create(),
           "boat": cv2.BRISK.create(),
           "graf": cv2.BRISK.create(),
           "leuven": cv2.BRISK.create(),
           "trees": cv2.BRISK.create(),
           "ubc": cv2.BRISK.create(),
           "wall": cv2.BRISK.create()
        },
        "FAST": {
           "bark": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "bikes": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "boat": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "graf": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "leuven": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "trees": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "ubc": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True),
           "wall": cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)
        },
        "GFTT": {
           "bark": cv2.GFTTDetector_create(),
           "bikes": cv2.GFTTDetector_create(),
           "boat": cv2.GFTTDetector_create(),
           "graf": cv2.GFTTDetector_create(),
           "leuven": cv2.GFTTDetector_create(),
           "trees": cv2.GFTTDetector_create(),
           "ubc": cv2.GFTTDetector_create(),
           "wall": cv2.GFTTDetector_create()
        },
        "KAZE": {
           "bark": cv2.KAZE_create(),
           "bikes": cv2.KAZE_create(),
           "boat": cv2.KAZE_create(),
           "graf": cv2.KAZE_create(),
           "leuven": cv2.KAZE_create(),
           "trees": cv2.KAZE_create(),
           "ubc": cv2.KAZE_create(),
           "wall": cv2.KAZE_create()
        },
        "MSER": {
           "bark": cv2.MSER_create(),
           "bikes": cv2.MSER_create(),
           "boat": cv2.MSER_create(),
           "graf": cv2.MSER_create(),
           "leuven": cv2.MSER_create(),
           "trees": cv2.MSER_create(),
           "ubc": cv2.MSER_create(),
           "wall": cv2.MSER_create()
        },
        "ORB": {
           "bark": cv2.ORB_create(),
           "bikes": cv2.ORB_create(),
           "boat": cv2.ORB_create(),
           "graf": cv2.ORB_create(),
           "leuven": cv2.ORB_create(),
           "trees": cv2.ORB_create(),
           "ubc": cv2.ORB_create(),
           "wall": cv2.ORB_create()
        },
        "HarrisLaplaceFeatureDetector": {
           "bark": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "bikes": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "boat": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "graf": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "leuven": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "trees": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "ubc": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create(),
           "wall": cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create()
        },
        "SIFT": {
           "bark": cv2.xfeatures2d_SIFT.create(),
           "bikes": cv2.xfeatures2d_SIFT.create(),
           "boat": cv2.xfeatures2d_SIFT.create(),
           "graf": cv2.xfeatures2d_SIFT.create(),
           "leuven": cv2.xfeatures2d_SIFT.create(),
           "trees": cv2.xfeatures2d_SIFT.create(),
           "ubc": cv2.xfeatures2d_SIFT.create(),
           "wall": cv2.xfeatures2d_SIFT.create()
        },
        "STAR": {
           "bark": cv2.xfeatures2d_StarDetector.create(),
           "bikes": cv2.xfeatures2d_StarDetector.create(),
           "boat": cv2.xfeatures2d_StarDetector.create(),
           "graf": cv2.xfeatures2d_StarDetector.create(),
           "leuven": cv2.xfeatures2d_StarDetector.create(),
           "trees": cv2.xfeatures2d_StarDetector.create(),
           "ubc": cv2.xfeatures2d_StarDetector.create(),
           "wall": cv2.xfeatures2d_StarDetector.create()
        },
        "SURF": {
           "bark": cv2.xfeatures2d_SURF.create(),
           "bikes": cv2.xfeatures2d_SURF.create(),
           "boat": cv2.xfeatures2d_SURF.create(),
           "graf": cv2.xfeatures2d_SURF.create(),
           "leuven": cv2.xfeatures2d_SURF.create(),
           "trees": cv2.xfeatures2d_SURF.create(),
           "ubc": cv2.xfeatures2d_SURF.create(),
           "wall": cv2.xfeatures2d_SURF.create()
        }
    }

feature_descriptors = \
    {
        "AKAZE": cv2.AKAZE_create(),
        "BRISK": cv2.BRISK_create(),
        "KAZE": cv2.KAZE_create(),
        "ORB": cv2.ORB_create(),
        "BOOST": cv2.xfeatures2d_BoostDesc.create(),
        "BRIEF": cv2.xfeatures2d_BriefDescriptorExtractor.create(),
        "DAISY": cv2.xfeatures2d_DAISY.create(),
        "FREAK": cv2.xfeatures2d_FREAK.create(),
        "LATCH": cv2.xfeatures2d_LATCH.create(),
        "LUCID": cv2.xfeatures2d_LUCID.create(),
        "SIFT": cv2.xfeatures2d_SIFT.create(),
        "SURF": cv2.xfeatures2d_SURF.create(),
        "VGG": cv2.xfeatures2d_VGG.create()
    }

detector_to_use = ["AKAZE", "BRISK", "KAZE", "ORB", "BRISK", "BRISK", "BRISK", "BRISK", "BRISK", "BRISK",
                   "SIFT", "SURF", "BRISK"]
feature_descriptors_to_test = ["AKAZE", "BRISK", "KAZE", "ORB", "BOOST", "BRIEF", "DAISY", "FREAK", "LATCH", "LUCID",
                               "SIFT", "SURF", "VGG"]

all_match_rates = {}
base_output_folder = "out/FeatureDescriptionTesting/"

for dataset_name in datasets:
    dataset = DatasetInfo(datasets[dataset_name])
    dataset_match_rates = {}
    for feature_detector, feature_descriptor in zip(detector_to_use, feature_descriptors_to_test):
        correct_match_count, match_count, images, non_max_list, thresh_list, num_features = \
            evaluate_detector_on_dataset(dataset,
                                         feature_detectors[feature_detector][dataset_name],
                                         feature_descriptors[feature_descriptor])
        dataset_match_rates[feature_detector + "-" + feature_descriptor] = {
            "correct_match_count": correct_match_count,
            "match_count": match_count,
            "Thresholds": thresh_list,
            "NonMax": non_max_list,
            "detected_count": num_features
        }
        _, ax = Plotter.get_zero_padded_fig_and_ax("img1_2")
        ax.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img1_3")
        ax.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img1_4")
        ax.imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img1_5")
        ax.imshow(cv2.cvtColor(images[3], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img1_6")
        ax.imshow(cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB))
        Plotter.save_plots(True, base_output_folder + dataset_name + "/" +
                           feature_detector + "_" + feature_descriptor + "/")
        Plotter.show_plots(False)
    all_match_rates[dataset_name] = dataset_match_rates

with open(base_output_folder + 'match_rates.json', 'w') as fp:
    json.dump(all_match_rates, fp, indent=4)
