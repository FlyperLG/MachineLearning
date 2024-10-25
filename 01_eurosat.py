import os
from PIL import Image
import numpy as np
import random

'''
Landscape Classification with the EUROSAT dataset
from Patrick Helber / DFKI.

> https://github.com/phelber/EuroSAT
'''

# the classes (= types of landscape)
# in the EUROSAT dataset
CLASSES = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}


def read_eurosat(base_path, n):
    '''
    Given a base directory base_path, this method reads the EUROSAT dataset
    from the folders below:

      base_path/AnnualCrop/AnnualCrop_1000.jpg
      base_path/AnnualCrop/AnnualCrop_1000.jpg
      ...
      base_path/SeaLake/SeaLake_99.jpg
      base_path/SeaLake/SeaLake_9.jpg

    Each JPEG contains a RGB image displaying a patch of landscape,
    which is converted into a 64 x 64 x 3 numpy array.

       @type base_path: string
       @param base_path: the base directory under which the EUROSAT data reside.

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
        for f in os.listdir(cpath)[:n]:
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


# to implement in Week 02
class KNNClassifier():

    def __init__(self, K):
        raise NotImplementedError()  # FIXME

    def train(self, X, y):
        '''train the classifier.

        @type X: np.array (N x D)
        @param X: the feature vectors (not the original images!) to train on,
                  each D-dimensional.

        @type y: np.array (N)
        @param y: the class labels (one per feature).
        '''
        raise NotImplementedError()  # FIXME

    def apply(self, x):
        ''' apply the classifier.

        @type x: np.array (D)
        @param x: the feature vector to classify.


        @rtype: tuple
        @return: returns a tuple (bestc,kidx), where bestc is the class with the most
                 votes and kidx are the indices of the best nearest neighbors in the
                 training data (ordered by ascending distance to the query).

                 Example: We query with x and obtain 3 nearest neighbors, training
                          training samples no. 14, 256 and 42. These training samples
                          belong to classes 3, 7 and 3 (i.e., class 3 obtains the most votes).

                          -> apply() returns (3, [14,256,42]).
        '''
        raise NotImplementedError()  # FIXME


class KNNClassifierAnnoy(KNNClassifier):
    ''' Like KNNClassifier, this class performs K-NN classification.
        However, a linear scan over the training data is replaced with
        a fast NN lookup using the annoy library: https://github.com/spotify/annoy
    . '''

    def __init__(self, K, ntrees):
        '''see KNNClassifier '''
        raise NotImplementedError()  # FIXME

    def train(self, X, y):
        '''see KNNClassifier '''
        raise NotImplementedError()  # FIXME

    def apply(self, x):
        '''see KNNClassifier '''
        raise NotImplementedError()  # FIXME


class FeatureExtractorFlatten():
    ''' this feature extractor simply flattens each 64 x 64 x 3 image
        into a (64*64*3)-dimensional feature vector x. '''

    def __init__(self):
        pass

    def apply(self, imgs):
        return [img.flatten() for img in imgs]


class FeatureExtractorColorHistogram():
    ''' this feature extractor computes a B x B x B color histogram
        for each input image. '''

    def __init__(self, B=8):
        # determine bins of color histogram
        self.bins = 3 * [np.arange(0, 257., 257. / B)]

    def apply(self, imgs):
        imgs_flat = [img.reshape(64 * 64, 3) for img in imgs]
        return [np.histogramdd(img, bins=self.bins)[0].flatten() for img in imgs_flat]


def plot_mosaic(image_grid, filename='neighbors.png'):
    '''
    plots the nearest neighbor of training images. It generates a grid of images
    (see the exercise sheet). Each test image corresponds to a row. In this row, the
    test image itself is placed on the left, followed by its K nearest neighbors
    from left to right. So, each row is a list of K+1 images, and the mosaic is a list
    of such rows.

    @type image_grid: list< list< (64 x 64 x 3 numpy array) > >
    @param image_grid: a list of rows, each a list of K+1 images, each a numpy array.

    @type filename: string
    @param filename: the filename where to store the mosaic image with neighbors.
    '''
    mosaic = []
    for row in image_grid:
        row = np.concatenate(row, axis=1)
        mosaic.append(row)
    mosaic = np.concatenate(mosaic, axis=0)
    mosaic = mosaic.astype(np.uint8)
    Image.fromarray(mosaic).save(filename)


# main program
if __name__ == "__main__":
    # read dataset.
    # FIXME: replace BASE directory if needed
    imgs, y = read_eurosat('EuroSAT_RGB', 100)
    print('Read EUROSAT dataset with %d samples (images).' % len(imgs))

    print("\n")

    # FIXME (Sheet 01): inspect the classes + average images of the dataset
    [print(f"{name}: {amount}") for name, amount in [(value, np.count_nonzero(y==key)) for key, value in CLASSES.items()]]

    # FIXME (Sheet 02): split training+test data

    # FIXME (Sheet 02): train the classifier

    # FIXME (Sheet 02): plot some nearest neighbors

    # FIXME (Sheet 02): run test, compute accuracy + confusion matrix
