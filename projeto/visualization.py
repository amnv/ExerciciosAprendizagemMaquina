import matplotlib.pyplot as plt

class Visualization:

    """ List of tuple [(hit_rate, k)] """
    @staticmethod
    def hit_rate_per_k(hit_list, k_list, file_name, weighted = False):

        plt.plot(k_list, hit_list)
        # title = "(knn com peso)" if weighted else "(Knn sem peso)"
        plt.title("Taxa de acerto por valor de k ")# + title)
        plt.xlabel('Valor de k')
        plt.ylabel("Taxa de acerto")

        plt.savefig(file_name)
        #plt.show()
        plt.close()
