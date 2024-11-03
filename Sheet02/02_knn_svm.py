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

        self.trainedModel = svm.SVC(C=1, kernel='rbf', gamma='auto')

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
        prediction = self.trainedModel.predict(x)
        return prediction


# main program
if __name__ == "__main__":

    # read dataset.
    imgs,y = read_eurosat('../EuroSAT_RGB', 1000)
    print('Read EUROSAT dataset with %d samples (images).' %len(imgs))

    # FIXME (Sheet 02): split training+test data
    imgs_train, imgs_valid, imgs_test, ytrain, yvalid, ytest = splits(imgs, y)

    Xtrain = feature_extraction(imgs_train)
    Xvalid = feature_extraction(imgs_valid)
    Xtest  = feature_extraction(imgs_test)

    # FIXME: enjoy coding ...

    # FIXME (Sheet 02): train the classifier
    # knnClassifier = KNNClassifier()
    # knnClassifier.fit(Xtrain, ytrain)
    # resultTrainKnn = [knnClassifier.predict(img) for img in Xtrain]
    # resultValidKnn = [knnClassifier.predict(img) for img in Xvalid]
    # resultTestKnn = [knnClassifier.predict(img) for img in Xtest]
    #
    # print('KNN Train:', accuracy(ytrain, resultTrainKnn))
    # print('KNN Valid:', accuracy(yvalid, resultValidKnn))
    # print('KNN Test:', accuracy(ytest, resultTestKnn))

    svmClassifier = SVMClassifier()
    svmClassifier.fit(Xtrain, ytrain)

    resultTrainSvm = svmClassifier.predict(Xtrain)
    # resultValidSvm = [svmClassifier.predict(img) for img in Xvalid]
    # resultTestSvm = [svmClassifier.predict(img) for img in Xtest]

    print(resultTrainSvm)

    print('SVM Train:', accuracy(ytrain, resultTrainSvm))
    # print('SVM Valid:', accuracy(yvalid, resultValidSvm))
    # print('SVM Test:', accuracy(ytest, resultTestSvm))
