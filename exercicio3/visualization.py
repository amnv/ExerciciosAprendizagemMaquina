import matplotlib.pyplot as plt

class Visualization:

    """ List of tuple [(hit_rate, k)] """
    @staticmethod
    def hit_rate_per_k(x_list, y_list, file_name):

        plt.plot(x_list, y_list)
        plt.title("Taxa de acerto por valor de k")
        plt.xlabel('Número de componentes principais')
        plt.ylabel("Taxa de acerto")

        plt.savefig(file_name)
        #plt.show()
        plt.close()
