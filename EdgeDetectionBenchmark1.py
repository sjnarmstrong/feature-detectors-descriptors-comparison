import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import makedirs
from os.path import dirname


save_plots = True
class Plotter:
    figure_dict = {}
    plt.interactive(False)

    @staticmethod
    def get_zero_padded_fig_and_ax(figure_name, figsize=[6, 6]):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        Plotter.figure_dict[figure_name] = fig
        return fig, ax

    @staticmethod
    def save_plots(should_save, save_dir, save_format=".pdf"):
        if not should_save:
            Plotter.figure_dict = {}
            return
        makedirs(save_dir, exist_ok=True)
        for img_name in Plotter.figure_dict:
            Plotter.figure_dict[img_name].savefig(
                save_dir + img_name + save_format, dpi=500, bbox_inches="tight", pad_inches=0)
        Plotter.figure_dict = {}

    @staticmethod
    def no_scale_save_plots(should_save, save_dir, img):
        if not should_save:
            return
        makedirs(dirname(save_dir), exist_ok=True)
        cv2.imwrite(save_dir, img)

    @staticmethod
    def show_plots(should_show):
        if should_show:
            plt.show()
        else:
            plt.close('all')

    @staticmethod
    def create_other(figure_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Plotter.figure_dict[figure_name] = fig
        return fig, ax


edge_mask = np.zeros((100, 100))
edge_mask[:, 49:51] = 1


trials_at_sigma = 300
sigma = 50
sigma_intervals = np.linspace(0, 120, 100)
# sigma_intervals = np.linspace(0, 50, 10)
# sigma_intervals = np.linspace(200, 200, 1)
correct_count_canny = np.zeros((len(sigma_intervals), trials_at_sigma), dtype=np.uint32)
incorrect_count_canny = np.zeros(correct_count_canny.shape, dtype=np.uint32)

correct_count_sobel = np.zeros((len(sigma_intervals), trials_at_sigma), dtype=np.uint32)
incorrect_count_sobel = np.zeros(correct_count_canny.shape, dtype=np.uint32)

correct_count_lap = np.zeros((len(sigma_intervals), trials_at_sigma), dtype=np.uint32)
incorrect_count_lap = np.zeros(correct_count_canny.shape, dtype=np.uint32)

for j, sigma in enumerate(sigma_intervals):
    print(str(j/len(sigma_intervals)*100)+"% Complete")
    for i in range(trials_at_sigma):

        noisy_image = np.clip(np.random.normal(sigma, sigma, edge_mask.shape), 0, 255).astype(np.uint8)
        noisy_image[:, :50] = np.clip(np.random.normal(255-sigma, sigma, (100, 50)), 0, 255).astype(np.uint8)

        edges_c = cv2.Canny(noisy_image, 40000, 55000, apertureSize=7, L2gradient=False)
        edges_in_mask = edges_c*edge_mask
        correct_count_canny[j, i] = np.count_nonzero(np.sum(edges_in_mask, axis=1))
        incorrect_count_canny[j, i] = np.count_nonzero(edges_c) - np.count_nonzero(edges_c*edge_mask)

        edges_l = np.abs(cv2.Laplacian(noisy_image, cv2.CV_64F)) > 230
        edges_in_mask = edges_l*edge_mask
        correct_count_lap[j, i] = np.count_nonzero(np.sum(edges_in_mask, axis=1))
        incorrect_count_lap[j, i] = np.count_nonzero(edges_l) - np.count_nonzero(edges_l*edge_mask)

        edges_s = (cv2.Sobel(noisy_image, cv2.CV_64F, 1, 0, ksize=7) ** 2 + cv2.Sobel(noisy_image, cv2.CV_64F, 0, 1,
                                                                                  ksize=7) ** 2) > 6658560000
        edges_in_mask = edges_s*edge_mask
        correct_count_sobel[j, i] = np.count_nonzero(np.sum(edges_in_mask, axis=1))
        incorrect_count_sobel[j, i] = np.count_nonzero(edges_s) - np.count_nonzero(edges_s*edge_mask)

        if i == 0 and save_plots:
            _, ax = Plotter.get_zero_padded_fig_and_ax("img_"+str(sigma))
            ax.imshow(noisy_image, cmap='gray')
            _, ax = Plotter.get_zero_padded_fig_and_ax("imgS_"+str(sigma))
            ax.imshow(edges_s, cmap='gray')
            _, ax = Plotter.get_zero_padded_fig_and_ax("imgC_"+str(sigma))
            ax.imshow(edges_c, cmap='gray')
            _, ax = Plotter.get_zero_padded_fig_and_ax("imgL_"+str(sigma))
            ax.imshow(edges_l, cmap='gray')

total_correct_in_mask = np.count_nonzero(np.sum(edge_mask, axis=1))
total_incorrect_in_mask = np.count_nonzero(edge_mask == 0)
avg_corr_canny = np.average(100*correct_count_canny/total_correct_in_mask, axis=1)
avg_incorr_canny = np.average(100*incorrect_count_canny/total_incorrect_in_mask, axis=1)
avg_corr_sob = np.average(100*correct_count_sobel/total_correct_in_mask, axis=1)
avg_incorr_sob = np.average(100*incorrect_count_sobel/total_incorrect_in_mask, axis=1)
avg_corr_lap = np.average(100*correct_count_lap/total_correct_in_mask, axis=1)
avg_incorr_lap = np.average(100*incorrect_count_lap/total_incorrect_in_mask, axis=1)

print(avg_corr_canny)
print(avg_incorr_canny)
print(avg_corr_sob)
print(avg_incorr_sob)
print(avg_corr_lap)
print(avg_incorr_lap)

plt.rc('text', usetex=True)
fig, ax = Plotter.create_other("avg_corr_edge_detection")
ax.plot(sigma_intervals, avg_corr_canny, color='r', lw=1, label="Canny")
ax.plot(sigma_intervals, avg_corr_sob, color='g', lw=1, label="Sobel")
ax.plot(sigma_intervals, avg_corr_lap, color='b', lw=1, label="Laplacian")
ax.set_title("Graph of Average TP Percentage vs. Image Noise")
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'TP (\%)')
ax.legend()

_, ax = Plotter.create_other("avg_incorr_edge_detection")
ax.plot(sigma_intervals, avg_incorr_canny, color='r', lw=1, label="Canny")
ax.plot(sigma_intervals, avg_incorr_sob, color='g', lw=1, label="Sobel")
ax.plot(sigma_intervals, avg_incorr_lap, color='b', lw=1, label="Laplacian")
ax.set_title("Graph of Average FP Percentage vs. Image Noise")
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'FP (\%)')
ax.legend()
Plotter.show_plots(not save_plots)
Plotter.save_plots(save_plots, "out/EdgeDetectionBenchmark1/")



ax.set_title("Graph of the Gamma Transformation Function")
ax.set_xlabel(r'p')
ax.set_ylabel(r'$$\alpha*p^\gamma$$')
Plotter.save_plots(True, "../out/log/")