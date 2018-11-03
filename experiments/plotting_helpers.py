import numpy as np


def get_running_means_w_std_bounds_and_legend_on_diff_prev_time_step(list_of_list_values):
    running_mean_and_std_bounds = []
    legend = ["Mean-STD", "Mean Difference", "Mean+STD"]
    for ii, losses_in_batch_ii in enumerate(list_of_list_values):
        if ii == 0:  # there's no previous time step to compare to
            running_mean_and_std_bounds.append([0, 0, 0])
        else:
            loss_diffs = [loss_val - list_of_list_values[ii - 1][jj]
                          for jj, loss_val in enumerate(losses_in_batch_ii)]
            mean = np.mean(loss_diffs)
            std = np.std(loss_diffs)

            running_mean_and_std_bounds.append([mean - std, mean, mean + std])

    return running_mean_and_std_bounds, legend


def get_running_means_w_std_bounds_and_legend(list_of_list_values):
    running_mean_and_std_bounds = []
    legend = ["Mean-STD", "Mean", "Mean+STD"]
    for ii in range(len(list_of_list_values)):
        mean = np.mean(list_of_list_values[ii])
        std = np.std(list_of_list_values[ii])

        running_mean_and_std_bounds.append([mean - std, mean, mean + std])

    return running_mean_and_std_bounds, legend


def get_running_std(list_of_list_values):
    return [np.std(list_of_list_values[ii]) for ii in range(len(list_of_list_values))]


def get_running_p_coeffs(list_of_list_values_1, list_of_list_values_2):
    assert len(list_of_list_values_1) == len(list_of_list_values_2)

    pearson_coefficients = []
    for ii in range(len(list_of_list_values_1)):
        cov = np.cov(np.stack((list_of_list_values_1[ii],
                               list_of_list_values_2[ii]), axis=0))[0, 1]
        std1 = np.std(list_of_list_values_1[ii])
        std2 = np.std(list_of_list_values_2[ii])
        correlation_coefficient = cov / (std1 * std2)

        pearson_coefficients.append(correlation_coefficient)

    return pearson_coefficients
