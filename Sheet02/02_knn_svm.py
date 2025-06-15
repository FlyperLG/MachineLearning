import time
import os
from PIL import Image
import numpy as np
import random
from annoy import AnnoyIndex
from sklearn import svm


# The classes (= types of landscape) in the EUROSAT dataset.
# We map each to an integer label.
CLASSES = {
    0 : 'AnnualCrop',
    1 : 'Forest',
    2 : 'HerbaceousVegetation',
    3 : 'Highway',
    4 : 'Industrial',
    5 : 'Pasture',
    6 : 'PermanentCrop',
    7 : 'Residential',
    8 : 'River',
    9 : 'SeaLake'
}


def read_eurosat(base_path, N=10000):
    '''
    Given a base directory 'base_path'', this method reads the EUROSAT dataset
    from the folders below:

      BASE/AnnualCrop/AnnualCrop_1000.jpg
      BASE/AnnualCrop/AnnualCrop_1000.jpg
      ...
      BASE/SeaLake/SeaLake_99.jpg
      BASE/SeaLake/SeaLake_9.jpg

    Each JPEG contains an RGB image displaying a patch of landscape,
    which is converted into a 64 x 64 x 3 numpy array.

       @type path: string
       @param path: the base directory under which the EUROSAT dataset resides.

       @rtype: tuple
       @return: The method returns a tuple (imgs,y), where
                imgs is a (N x 64 x 64 x 3) numpy array containing
                all images and y is a (N) numpy array containing
                the class labels, each a value from {0,....,9}.
    '''

    imgs = []
    y = []

    for label, name in CLASSES.items():

        cpath = base_path + os.sep + name
        assert os.path.exists(cpath)
        print('Reading directory', cpath, '...')

        # loop over a classes' images
        for f in os.listdir(cpath)[:N]:
            ipath = cpath + os.sep + f
            try:
                img = np.array(Image.open(ipath))
            except:
                print('Malformed image data in', ipath)
                exit(1)
            imgs.append(img)
            y.append(label)

    # shuffle data (zip->shuffle->unzip)
    _ = list(zip(imgs, y))
    random.shuffle(_)
    imgs, y = zip(*_)

    # turn lists into numpy arrays
    imgs = np.stack(imgs, axis=0)
    y = np.array(y)

    return (imgs, y)



class Classifier():

    def __init__(self):
        ''' constructor '''
        pass

    def fit(self, X, y):
        '''
        train the classifier and store the trained model as an attribute.

        @type X: np.array (N x D, dtype=float)
        @param X: the feature vectors to train on,
                  each D-dimensional.

        @type y: np.array (N, dtype=int)
        @param y: the class labels.
        '''
        raise NotImplementedError()

    def predict(self, x):
        '''
        apply the classifier to a new object.

        @type x: np.array (D, dtype=float)
        @param x: the feature vector to classify.

        @rtype: int
        @return: returns the predicted class.
        '''
        raise NotImplementedError()



def feature_extraction(imgs, nbins=8):
    '''
    turns each color image in 'imgs' into a color histogram to be used as a feature vector.
    there's no need to modify this.

    @param imgs: (N x 64 x 64 x 3) numpy array containing all images
                 (as returned by read_eurosat())
    @param nbins: number of bins per color channel

    @return: (N x D) numpy array containing the color histograms
    '''
    bins = 3*[np.arange(0, 257., 257./nbins)]
    imgs_flat = [img.reshape(64*64,3) for img in imgs]
    return np.array([np.histogramdd(img,bins=bins)[0].flatten() for img in imgs_flat])


def splits(imgs, y, nvalid=1000, ntest=1000):
    '''
    split dataset into training, validation and test split.
    '''
    n = len(y)
    ntrain = n - nvalid - ntest
    assert ntrain >= 100 # make sure we have enough training data

    imgs_train = imgs[:ntrain]
    imgs_valid = imgs[ntrain:ntrain+nvalid]
    imgs_test  = imgs[ntrain+nvalid:]
    ytrain = y[:ntrain]
    yvalid = y[ntrain:ntrain+nvalid]
    ytest  = y[ntrain+nvalid:]

    return imgs_train, imgs_valid, imgs_test, ytrain, yvalid, ytest


def accuracy(y, ypred):
    '''
    computes the percentage of correct classifications.

    @param y: numpy array of shape (N,) containing ground truth labels
    @param ypred: numpy array of shape (N,) containing the model's predictions
    @return: percentage of correct classifications (float).
    '''
    return np.mean(y == ypred)

class KNNClassifier(Classifier):
    trainedModel = []

    def fit(self, X, y):
        '''
        train the classifier and store the trained model as an attribute.

        @type X: np.array (N x D, dtype=float)
        @param X: the feature vectors to train on,
                  each D-dimensional.

        @type y: np.array (N, dtype=int)
        @param y: the class labels.
        '''
        start_time = time.time()
        self.trainedModel = list(zip(X, y))

        tree = AnnoyIndex(len(X[0]), 'manhattan')
        for i in range(0, len(self.trainedModel)):
            v = X[i]
            tree.add_item(i, v)
        tree.build(10)
        tree.save('knn.ann')

        end_time = time.time()

        print(f"Elapsed time training knn: {end_time - start_time} seconds")

        #distances = [[np.sum(abs(img1 - img2)) for img2 in X] for img1 in X]
        #print(distances)
        #sortedDistances = sorted(distances)
        #distance = np.sum(abs(X[0] - X[0]))
        #print(sortedDistances)




    def predict(self, x, n = 50):
        '''
        apply the classifier to a new object.

        @type x: np.array (D, dtype=float)
        @param x: the feature vector to classify.

        @rtype: int
        @return: returns the predicted class.
        '''
        u = AnnoyIndex(len(x), 'manhattan')
        u.load('knn.ann') # super fast, will just mmap the file
        neighbours = u.get_nns_by_vector(x, n)

        #distances = sorted([(np.sum(abs(x - z[0])), z[1]) for z in self.trainedModel])
        unique, counts = np.unique([self.trainedModel[n][1] for n in neighbours], return_counts=True)
        result = sorted(zip(counts, unique), reverse=True)
        return result[0][1]
        

class SVMClassifier(Classifier):
    trainedModel = []

    # def __init__(self):
    #     ''' constructor '''
    #     self.trainedModel =

    def fit(self, X, y, C = 1):
        '''
        train the classifier and store the trained model as an attribute.

        @type X: np.array (N x D, dtype=float)
        @param X: the feature vectors to train on,
                  each D-dimensional.

        @type y: np.array (N, dtype=int)
        @param y: the class labels.
        ''',
        start_time = time.time()

        subset_size = 500
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[indices]
        gamma = np.sum((X_subset[:, np.newaxis] - X_subset) ** 2)
        gamma = 1 / (gamma / (subset_size ** 2))
        print("Gamma: ", gamma)

        self.trainedModel = svm.SVC(C=C, kernel='rbf', gamma=gamma)

        self.trainedModel.fit(X, y)
        end_time = time.time()

        print(f"Elapsed time training svm: {end_time - start_time} seconds")

    def predict(self, x):
        '''
        apply the classifier to a new object.

        @type x: np.array (D, dtype=float)
        @param x: the feature vector to classify.

        @rtype: int
        @return: returns the predicted class.
        '''
        return self.trainedModel.predict([x])[0]

def train_classifier(Xtrain, ytrain, Xtest, ytest):
    knnClassifier = KNNClassifier()
    knnClassifier.fit(Xtrain, ytrain)
    resultTrainKnn = [knnClassifier.predict(img) for img in Xtrain]
    resultTestKnn = [knnClassifier.predict(img) for img in Xtest]

    print('KNN Train:', accuracy(ytrain, resultTrainKnn))
    print('KNN Test:', accuracy(ytest, resultTestKnn))
    print()

    svmClassifier = SVMClassifier()
    svmClassifier.fit(Xtrain, ytrain)

    resultTrainSvm = [svmClassifier.predict(img) for img in Xtrain]
    resultTestSvm = [svmClassifier.predict(img) for img in Xtest]

    print('SVM Train:', accuracy(ytrain, resultTrainSvm))
    print('SVM Test:', accuracy(ytest, resultTestSvm))

def grid_seaarch_knn(Xtrain, ytrain, Xvalid, yvalid):
    knnClassifier = KNNClassifier()
    knnClassifier.fit(Xtrain, ytrain); 

    knnResultDict = dict()
    k = 1
    for i in range(0, 20):
        resultValidKnn = [knnClassifier.predict(img, k) for img in Xvalid]
        knnResultDict[k] = accuracy(yvalid, resultValidKnn)
        k+=2
    return knnResultDict

def grid_seaarch_svm(Xtrain, ytrain, Xvalid, yvalid):
    svmClassifier = SVMClassifier()
    svmResultDict = dict()
    C = 1/10000
    for i in range(0, 15):
        svmClassifier.fit(Xtrain, ytrain, C)
        resultValidSvm = [svmClassifier.predict(img) for img in Xvalid]
        svmResultDict[C] = accuracy(yvalid, resultValidSvm)
        C *= 10
    return svmResultDict

# main program
if __name__ == "__main__":

    # read dataset.
    imgs,y = read_eurosat('Sheet01/EuroSAT_RGB', 500)
    print('Read EUROSAT dataset with %d samples (images).' %len(imgs))

    # FIXME (Sheet 02): split training+test data
    imgs_train, imgs_valid, imgs_test, ytrain, yvalid, ytest = splits(imgs, y)

    Xtrain = feature_extraction(imgs_train)
    Xvalid = feature_extraction(imgs_valid)
    Xtest  = feature_extraction(imgs_test)

    # FIXME: enjoy coding ...

    # FIXME (Sheet 02): train the classifier
    # result = grid_seaarch_svm(Xtrain, ytrain, Xvalid, yvalid)
    result = grid_seaarch_knn(Xtrain, ytrain, Xvalid, yvalid)
    print(result)
