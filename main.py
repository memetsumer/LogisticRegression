import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_breast_cancer

def main():
    dataset = load_breast_cancer()

    X = dataset.data[:300, np.arange(1, 30, 1)]
    y = dataset.target[:300]

    lr = LogisticRegression(X, y, 0.00008, 0.00007)

    for i in range(500):
        lr.gradient()

    unseenX = dataset.data[300:, np.arange(1, 30, 1)]
    actualY = dataset.target[300:]

    score = 0
    for i, j in zip(unseenX, actualY):
        value = None

        if lr.predict(i) >= 0.00000001:
            value = 1

        elif lr.predict(i) < 0.00000001:
            value = 0

        if value == j:
            score += 1
        print(f"actual y: {j}   predicted y: {value}    score: {score}    accuracy: {score / 259}")


if __name__ == '__main__':
    main()