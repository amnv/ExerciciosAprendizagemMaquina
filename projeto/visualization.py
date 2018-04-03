import matplotlib.pyplot as plt

class Visualization:

    """ List of tuple [(hit_rate, k)] """
    @staticmethod
    def hit_rate_per_k(data, weighted = False, file_name = 'books_read.png'):
        hit_list = []
        k_list = []

        for hit_rate, k in data:
            hit_list.append(hit_rate)
            k_list.append(k)

        plt.plot(hit_list, k_list)
        title = "(knn com peso)" if weighted else "(Knn sem peso)"
        plt.title("Taxa de acerto por valor de k " + title)
        plt.xlabel('Valor de k')
        plt.ylabel("Taxa de acerto")

        plt.savefig(file_name)
        #plt.show()



def main():
    a = [(1,2), (10,3), (0,41)]
    Visualization.hit_rate_per_k(a, "asd")

main()