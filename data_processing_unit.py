import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
    Data = dict()
    for i in range(1, 10):
        Data[i] = []
    f = open(filename, 'r')
    for line in f:
        line = line.strip().split(' ')
        for x in range(1,11):
            if np.abs(float(line[0]) - x) < 0.1:
                picture = np.zeros((16,16))
                for i in range(1, len(line)):
                    picture[(i-1) // 16, (i - 1) % 16] = float(line[i])
                Data[int(float(line[0]))].append(picture)
    return Data

def XandY(train, test):
    X = list()
    y = list()
    for n in train.keys():
        if n==1:
            classification = 1
        else:
            classification = -1
        for x in range(len(train[n])):
            X.append(train[n][x])
            y.append(classification)
        for x in range(len(test[n])):
            X.append(test[n][x])
            y.append(classification)
    return X, y

def calculate_symmetry(image):
    vertical_symmetry = np.mean(np.abs(image - np.flipud(image)))
    horizontal_symmetry = np.mean(np.abs(image - np.fliplr(image)))
    return vertical_symmetry, horizontal_symmetry

def find_transform(data, transform):
    X = []
    for i, image in enumerate(data):
        X.append(transform(image))
    return X

def normalize_data(data, y):
    t1_min = float("inf")
    t1_max = 0
    t2_min = float("inf")
    t2_max = 0
    X = []
    for i, t in enumerate(data):
        if t[0]<t1_min:
            t1_min = t[0]
        elif t[0]>t1_max:
            t1_max = t[0]
        if t[1]<t2_min:
            t2_min = t[1]
        elif t[1]>t2_max:
            t2_max = t[1]

    for i, t in enumerate(data):
        n1 = 2*(data[i][0]-t1_min) / (t1_max-t1_min)-1
        n2 = 2*(data[i][1] - t2_min) / (t2_max-t2_min)-1
        n = np.array((n1, n2, y[i]))
        
        X.append(n)
    X = np.array(X)
    return X
def create_and_remove_test_data(data, num_samples=150):
    np.random.seed(2)
    X = []
    selected_indices = np.random.choice(len(data), num_samples, replace=False)
    selected_data = np.array([data[i] for i in selected_indices])
    rems = np.array([data[i] for i in range(len(data)) if i not in selected_indices])
    return selected_data, rems


if __name__=="__main__":

    train = get_data("train.txt")

    test  = get_data('test.txt')

    X, y= XandY(train, test)
    X = find_transform(X, calculate_symmetry)
    X = normalize_data(X, y)
    X,test = create_and_remove_test_data(X)
    y = X[:, 2]
    y_test = test[:, 2]
    
    np.savetxt("data.txt", X, fmt='%1.4f')


