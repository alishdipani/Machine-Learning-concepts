import numpy as np


def L2Loss(y_predicted, y_ground_truth, reduction="None"):
    """returns l2 loss between two arrays

    :param y_predicted: array of predicted values
    :type y_predicted: ndarray
    :param y_ground_truth: array of ground truth values
    :type y_ground_truth: ndarray
    :param reduction: reduction mode, defaults to "mean"
    :type reduction: str, optional
    :return: l2-loss
    :rtype: scalar if reduction is sum or mean, else ndarray
    """
    # Calculate the difference array
    difference = y_predicted - y_ground_truth
    # Raise every difference value to the power of 2
    squared_difference = np.multiply(difference, difference)
    # L2 distance is the reduced form of the squared difference array
    if reduction == "sum":
        # Reduction can be done by summing up all the values in the difference array (this is known as "L2-Loss")
        l2_distance = np.sum(squared_difference)
        return l2_distance
    elif reduction == "mean":
        # Reduction can also be done by taking the mean (this is known as "Mean Squared Error")
        mean_squared_error = np.mean(squared_difference)
        return mean_squared_error
    elif reduction == "None":
        return squared_difference
    else:
        print('ValueError: reduction should be "sum" / "mean" / "None"')


def main():
    print("Initializing predicted and ground truth arrays:\n")
    print('(NOTE: Enter the values in a space-separated format. Ex: "5.36 1.02 2.03")')
    y_predicted = [
        float(item) for item in input("Enter the predicted values: ").split()
    ]
    y_ground_truth = [
        float(item)
        for item in input("Enter the corresponding ground truth values: ").split()
    ]
    assert len(y_predicted) == len(
        y_ground_truth
    ), "Number of predicted values {} and ground truth {} values should match".format(
        len(y_predicted), len(y_ground_truth)
    )
    y_predicted = np.array(y_predicted)
    y_ground_truth = np.array(y_ground_truth)
    reduction = str(input('Enter the reduction mode: "sum" / "mean" / "None": '))
    loss = L2Loss(y_predicted, y_ground_truth, reduction=reduction)
    print("L2-Loss with {}-reduction: {}".format(reduction, loss))


if __name__ == "__main__":
    main()
