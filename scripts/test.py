import numpy as np




if __name__ == "__main__":

    arr1 = np.array([1, 2, 3])

    arr2 = np.array([4, 5, 6])

    arr = np.vstack((arr1, arr2))

    print(arr)