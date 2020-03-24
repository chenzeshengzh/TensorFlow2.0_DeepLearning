import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = ['STKaiti']
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['axes.unicode_minus'] = False

# track loss with epoch
loss_with_step = []


def mse(b, w, points):
    """
    calculate mean square error
    :param b: bias
    :param w: weights
    :param points: input like (x, y)
    :return: mse value
    """
    total_error = 0
    for i in range(len(points)):
        x, y = points[i, :]
        total_error += (y - (w * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, w_current, points, lr):
    """
    calculate grads in each iteration
    :param b_current: b before this iteration
    :param w_current: w before this iteration
    :param points: input like (x, y)
    :param lr: learning rate
    :return: grads for (b, w) in each iteration
    """
    # initialize w and b
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))

    # calculate grads
    for i in range(0, len(points)):
        x, y = points[i, :]
        b_gradient += (2 / M) * ((w_current * x + b_current) - y)
        w_gradient += (2 / M) * ((w_current * x + b_current) - y) * x

    # refresh w and b
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    """
    gradient descent in some iterations
    :param points: input like (x, y)
    :param starting_b: initialize b
    :param starting_w: initialize w
    :param lr: learning rate
    :param num_iterations: max iterations
    :return: result for (b, w)
    """
    # get initialize params
    b = starting_b
    w = starting_w

    # start gradient iterations
    for step in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points)
        loss_with_step.append(loss)
        if step % 50 == 0:
            print(f'iteration:{step}, loss:{loss}, w:{w}, b:{b}')
    return [b, w]


def main():
    # sample (x, y) in the model
    data = []
    for i in range(100):
        # U distribution for x
        x = np.random.uniform(-10., 10.)
        # N distribution for noise
        eps = np.random.normal(0., 0.01)
        # get y using the model and plus noise
        y = 1.477 * x + 0.089 + eps
        data.append([x, y])
    data = np.array(data)

    # initialize params
    lr = 1e-2
    initial_b, initial_w = 0, 0
    num_iterations = 1000

    # predict w and b
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w:{w}, b:{b}')

    # plot loss with step(epoch)
    plt.figure()
    plt.plot(np.arange(len(loss_with_step)), loss_with_step, color='C0', label='训练误差')
    plt.plot(np.arange(len(loss_with_step)), loss_with_step, 's')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE - Epochs')
    plt.show()


if __name__ == '__main__':
    main()
