import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class dA(object):
    
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        """Create a Theano random generator that gives symbolic random values less than 2**30"""
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        """ W is initialized with `initial_W` which is uniformely sampled from low to high. The output of uniform is converted using asarray to 
            dtype theano.config.floatX so that the code is runable on GPU
        """
        if not W:
            
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        """Theano variables pointing to a set of biases values"""
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        
        self.W = W
        """ b is the bias for transition from x to y i.e bias of hidden variables """
        self.b = bhid
        
#       """ b' is the bias for transition from y to z i.e bias of visible variables """
        self.b_prime = bvis

        """ Here I am using tied weights and hence W_prime is just the transpose of W """
        self.W_prime = self.W.T

        self.theano_rng = theano_rng
        
        """ If input is not specified, then we self generate a particular matrix which will be serving as our input """
        
        if input is None:
            
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        """ If we were not using tied weights, we would have been including self.W_prime as a parameter too """
        self.params = [self.W, self.b, self.b_prime]

        
        """ I have applied Masking Noise(MN) corruption process where a fraction(corruption_level) of elements are chosen at randomnly and made 0 
        Note : First argument of theano.rng.binomial is the shape(size) of random numbers that it should produce, second argument is the number of 
           trials, third argument is the probability of success of any trial. This will produce an array of 0s and 1s where 1 has a probability of
           1 - ``corruption_level`` and 0 with ``corruption_level``

           The binomial function return int64 data type by default.  int64 multiplicated by the input type(floatX) always return float64. 
           To keep all data in floatX when floatX is float32, we set the dtype of the binomial to floatX. As in our case the value of the
           binomial is always 0 or 1, this don't change the
           result. This is needed to allow the gpu to work correctly as it only support float32 for now.
        """
    def get_corrupted_input(self, input, corruption_level):
    
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) * input

    
    """ To compute the values present in the hidden layer(y) """
    def get_hidden_values(self, input):
        
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    """ To compute the reconstructed input(z) from y """
    def get_reconstructed_input(self, hidden):
        
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    """ This function computes the cost and the updates for one training step of the dA """
    def get_cost_updates(self, corruption_level, learning_rate):
        

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):

    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # Compute number of minibatches for training, validation and testing
    # Without borrow=True, this will copy the shared variable content. To remove the copy you can use the borrow parameter like this:
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    """ BUILDING THE MODEL WITH 0% CORRUPTION """

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    """ TRAINING """
    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    """ BUILDING WITH CORRUPTION 30% """

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=1,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    """ Training """

    # go through training epochs
    for epoch in range(training_epochs):
        # go through trainng set
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The 100% corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
    

    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')
    

    os.chdir('../')


if __name__ == '__main__':
    test_dA()