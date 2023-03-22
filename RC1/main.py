import sys
import pandas as pd
from svd import SVD

def main():
    trainPath = sys.argv[1] if len(sys.argv) > 1 else "ratings.csv"
    targetPath = sys.argv[2] if len(sys.argv) > 2 else "targets.csv"

    train = pd.read_csv(trainPath)
    target = pd.read_csv(targetPath)

    svd  = SVD(k = 100, alpha = 0.1, regularization = 0.02, epochs=20, seed=0)
    svd.fit(train)
    predictions = svd.predictTargets(target)

    print("UserId:ItemId,Rating")
    for i in predictions:
        print(f'{i[0]},{i[1]}')




if __name__ == "__main__":
    main()
