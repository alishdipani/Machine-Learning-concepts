import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_graph(x, save_graph=False):
    plt.plot(x, sigmoid(x))
    if save_graph:
        plt.savefig("sigmoid.jpg")
    else:
        plt.show()


def main():
    print("Generating graph")
    save_graph = False  # save the graph (true) or show (false)
    generate_graph(np.linspace(-10, 10, 100), save_graph=save_graph)


if __name__ == "__main__":
    main()
