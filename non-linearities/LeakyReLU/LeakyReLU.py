import numpy as np
import matplotlib.pyplot as plt


def LeakyReLU(x):
    return np.where(x>0, x, x*0.01)



def generate_graph(x, save_graph=False):
    plt.plot(x, LeakyReLU(x))
    if save_graph:
        plt.savefig("LeakyReLU.jpg")
    else:
        plt.show()


def main():
    print("Generating graph")
    save_graph = False  # save the graph (true) or show (false)
    generate_graph(np.linspace(-200, 50, 10), save_graph=save_graph)


if __name__ == "__main__":
    main()
