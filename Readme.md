                                           ##################################
                                           ######DENOISING AUTOENCODERS######
                                           ##################################
                                           
 This is a version of Denoising autoencoders which runs for three corruption levels- 0%, 30% and 100%.
 
 The basic ideology behing autoencoders is to train the autoencoder to reconstruct the input from a corrupted version of it in order to force the hidden layer to discover more robust features and prevent it from simply learning the identity.
 
 An autoencoder takes an input \mathbf{x} \in [0,1]^d and first maps it (with an encoder) to a hidden representation \mathbf{y} \in [0,1]^{d'} through a deterministic mapping, e.g.:

\mathbf{y} = s(\mathbf{W}\mathbf{x} + \mathbf{b})

Where s is a non-linearity such as the sigmoid. The latent representation \mathbf{y}, or code is then mapped back (with a decoder) into a reconstruction \mathbf{z} of the same shape as \mathbf{x}. The mapping happens through a similar transformation, e.g.:

\mathbf{z} = s(\mathbf{W'}\mathbf{y} + \mathbf{b'})

(Here, the prime symbol does not indicate matrix transposition.) \mathbf{z} should be seen as a prediction of \mathbf{x}, given the code \mathbf{y}. 

The reconstruction error can be measured in many ways, depending on the appropriate distributional assumptions on the input given the code. The traditional squared error L(\mathbf{x} \mathbf{z}) = || \mathbf{x} -
\mathbf{z} ||^2, can be used. If the input is interpreted as either bit vectors or vectors of bit probabilities, cross-entropy of the reconstruction can be used:

L_{H} (\mathbf{x}, \mathbf{z}) = - \sum^d_{k=1}[\mathbf{x}_k \log
        \mathbf{z}_k + (1 - \mathbf{x}_k)\log(1 - \mathbf{z}_k)]
        
Some salient features of this implementation are:

* I have applied Masking Noise(MN) corruption process where a fraction(corruption_level) of elements are chosen at randomnly and made 0.
  Other corruption process which could have been used are
      i) Salt and Pepper Noise - A fraction ν of the elements of x (chosen at random for each example) is set to their minimum or maximum possible value (typically 0 or 1) according to a fair coin flip.
      ii)Additive Isotropic Gaussian Noise - x ̃|x ∼ N (x,σ2I); 
 
* This runs for three different corruption levels- 0%, 30% and 100%. The plots for these can be seen in dA_plots.
* Implemenation is done using Theano using a class so that later it can be used to construct a stacked autoencoder.
* I have used tied weights i.e. the weight matrix W' is made to be the transpose of the weight matrixx W (\mathbf{W'} = \mathbf{W}^T).
* The weights of the dA class have to be shared with those of a corresponding sigmoid layer. For this reason, the constructor of the dA also gets Theano variables pointing to the shared parameters. If those parameters are left to None, new ones will be constructed.


Implementation is done using 
* Python 3.5.2 :: Anaconda 4.2.0
* MNIST Dataset(mnist.pkl.gz)

Requirements
* Theano
* Numpy
* Python Image Library - To save the filters as an image

