import cv2
import numpy as np
from plotTools import Plotter
import json


def get_match_rate(kp_1_np, kp_2_np, T_1_2, img2_shape, thresh_sq):
    kp_T_1_2 = np.dot(T_1_2, kp_1_np.T)
    kp_T_1_2 = (kp_T_1_2 / kp_T_1_2[2]).T
    points_out_of_frame = np.count_nonzero(np.logical_or(kp_T_1_2[:, :2] >= img2_shape[:2],
                                                         kp_T_1_2[:, :2] < [0, 0]), axis=1)
    where_points_in_frame = np.where(points_out_of_frame == 0)
    kp_T_1_2 = kp_T_1_2[where_points_in_frame[0]]

    if len(kp_T_1_2)==0 or len(kp_2_np)==0:
        return float('inf')

    point_closer_than_thresh = np.sum(np.square(kp_2_np[:, None] -
                                                kp_T_1_2[None, :]), axis=2) < thresh_sq
    number_of_points_matched = np.count_nonzero(np.count_nonzero(point_closer_than_thresh, axis=0))
    match_rate = number_of_points_matched / len(kp_T_1_2)
    return match_rate


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
            print("Images not found in: "+dataset_path)
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
    current_step_size = f.getThreshold()/10
    is_int_thresh = type(f.getThreshold()) == int
    for i in range(max_iterations):
        if len(kp) < number_of_points_min:
            if current_step_positive_dir:
                current_step_size = current_step_size/10
            new_thresh = f.getThreshold() - current_step_size
            if is_int_thresh:
                new_thresh = int(new_thresh)
                if f.getThreshold() == new_thresh:
                    break
            f.setThreshold(new_thresh)
            current_step_positive_dir = False
        elif len(kp) > number_of_points_max:
            if current_step_positive_dir == False:
                current_step_size = current_step_size/10
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


def evaluate_detector_on_dataset(dataset: DatasetInfo, _filter, thresh_sq=9):
    kp1, non_max_1, thresh_1 = detect_key_points(_filter, dataset.img1)
    kp2, non_max_2, thresh_2 = detect_key_points(_filter, dataset.img2)
    kp3, non_max_3, thresh_3 = detect_key_points(_filter, dataset.img3)
    kp4, non_max_4, thresh_4 = detect_key_points(_filter, dataset.img4)
    kp5, non_max_5, thresh_5 = detect_key_points(_filter, dataset.img5)
    kp6, non_max_6, thresh_6 = detect_key_points(_filter, dataset.img6)
    kp_1_np = np.array(list(map(lambda x: x.pt + (1.0,), kp1)))
    kp_2_np = np.array(list(map(lambda x: x.pt + (1.0,), kp2)))
    kp_3_np = np.array(list(map(lambda x: x.pt + (1.0,), kp3)))
    kp_4_np = np.array(list(map(lambda x: x.pt + (1.0,), kp4)))
    kp_5_np = np.array(list(map(lambda x: x.pt + (1.0,), kp5)))
    kp_6_np = np.array(list(map(lambda x: x.pt + (1.0,), kp6)))
    
    mach_rate_1_2 = get_match_rate(kp_1_np, kp_2_np, dataset.t_1_2, dataset.img2.shape, thresh_sq)
    mach_rate_1_3 = get_match_rate(kp_1_np, kp_3_np, dataset.t_1_3, dataset.img3.shape, thresh_sq)
    mach_rate_1_4 = get_match_rate(kp_1_np, kp_4_np, dataset.t_1_4, dataset.img4.shape, thresh_sq)
    mach_rate_1_5 = get_match_rate(kp_1_np, kp_5_np, dataset.t_1_5, dataset.img5.shape, thresh_sq)
    mach_rate_1_6 = get_match_rate(kp_1_np, kp_6_np, dataset.t_1_6, dataset.img6.shape, thresh_sq)

    img1_features = dataset.img1.copy()
    img2_features = dataset.img2.copy()
    img3_features = dataset.img3.copy()
    img4_features = dataset.img4.copy()
    img5_features = dataset.img5.copy()
    img6_features = dataset.img6.copy()
    cv2.drawKeypoints(dataset.img1, kp1, img1_features, color=(0, 0, 255))
    cv2.drawKeypoints(dataset.img2, kp2, img2_features, color=(0, 0, 255))
    cv2.drawKeypoints(dataset.img3, kp3, img3_features, color=(0, 0, 255))
    cv2.drawKeypoints(dataset.img4, kp4, img4_features, color=(0, 0, 255))
    cv2.drawKeypoints(dataset.img5, kp5, img5_features, color=(0, 0, 255))
    cv2.drawKeypoints(dataset.img6, kp6, img6_features, color=(0, 0, 255))

    return ([mach_rate_1_2, mach_rate_1_3, mach_rate_1_4, mach_rate_1_5, mach_rate_1_6],
            [img1_features, img2_features, img3_features, img4_features, img5_features, img6_features],
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

all_match_rates = {}
base_output_folder = "out/FeatureDetectionTesting/"

for dataset_name in datasets:
    dataset = DatasetInfo(datasets[dataset_name])
    _, ax = Plotter.get_zero_padded_fig_and_ax("img1")
    ax.imshow(cv2.cvtColor(dataset.img1, cv2.COLOR_BGR2RGB))
    _, ax = Plotter.get_zero_padded_fig_and_ax("img2")
    ax.imshow(cv2.cvtColor(dataset.img2, cv2.COLOR_BGR2RGB))
    _, ax = Plotter.get_zero_padded_fig_and_ax("img3")
    ax.imshow(cv2.cvtColor(dataset.img3, cv2.COLOR_BGR2RGB))
    _, ax = Plotter.get_zero_padded_fig_and_ax("img4")
    ax.imshow(cv2.cvtColor(dataset.img4, cv2.COLOR_BGR2RGB))
    _, ax = Plotter.get_zero_padded_fig_and_ax("img5")
    ax.imshow(cv2.cvtColor(dataset.img5, cv2.COLOR_BGR2RGB))
    _, ax = Plotter.get_zero_padded_fig_and_ax("img6")
    ax.imshow(cv2.cvtColor(dataset.img6, cv2.COLOR_BGR2RGB))
    Plotter.save_plots(True, base_output_folder+dataset_name+"/images/")
    Plotter.show_plots(False)
    dataset_match_rates = {}
    for fd in feature_detectors:
        match_rates, images, non_max_list, thresh_list, num_features = evaluate_detector_on_dataset(
            dataset, feature_detectors[fd][dataset_name])
        dataset_match_rates[fd] = {
            "MatchRates": match_rates,
            "Thresholds": thresh_list,
            "NonMax": non_max_list,
            "detected_count": num_features
        }
        _, ax = Plotter.get_zero_padded_fig_and_ax("img1")
        ax.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img2")
        ax.imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img3")
        ax.imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img4")
        ax.imshow(cv2.cvtColor(images[3], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img5")
        ax.imshow(cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB))
        _, ax = Plotter.get_zero_padded_fig_and_ax("img6")
        ax.imshow(cv2.cvtColor(images[5], cv2.COLOR_BGR2RGB))
        Plotter.save_plots(True, base_output_folder+dataset_name+"/"+fd+"/")
        Plotter.show_plots(False)
    all_match_rates[dataset_name] = dataset_match_rates


with open(base_output_folder+'match_rates.json', 'w') as fp:
    json.dump(all_match_rates, fp, indent=4)
