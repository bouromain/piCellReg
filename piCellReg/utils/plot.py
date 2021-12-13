import matplotlib.pyplot as plt


def plot_session(s):

    m1 = s._iscell.astype(bool)

    plt.figure(figsize=(20, 20))
    plt.imshow(s._mean_image_e)

    plt.plot(s._x_center[m1], s._y_center[m1], "+r", ms=2)
