# An illustrated introduction to the t-SNE algorithm

INTRO

* complex, high-dimensional data
* need to visualize
* hard to visualize more than 3 dimensions
* sometimes, high-dimensional data only contains a few relevant dimensions
* automatically discovering this structure = manifold learning
* many algorithms exist
* one of the state-of-the-art algo is t-SNE, invented in 2008 by ** and Hinton
* it has been used in ** and **
* it is implemented in scikit-learn, the leading machine learning platforms in python
* in this post, we'll introduce and illustrate the main mathematical ideas underlying the algorithm
* we'll be using a famous data example throughout this post: the handwritten digits dataset (put screenshot). number of features? number of samples?
* demo: otuput of t-SNE on the dataset

The ancestor: SNE

* data points = original points in R^D
* map points = target points in R^2
* every data point has a corresponding map point, and conversely (bijection)
* we want the map points to reflect the structure of the data points
* how to do this? we want the distance to be kept
* here is how we express this idea
* we start from the distance matrix
* now, we consider a Gaussian distribution centered on each data point, with a given variance
* idem for the map points, but with a fixed variance
* we define a similarity matrix for the data points and the map points: sim(i,j) is roughly speaking the proba that j belongs to distrib i. close = high, far = low
* screenshot of the sim matrix
* idem for the map points: we want the two sim matrices to be close
* physical analogy: n-body problem with springs and strength (stifness?) depending on the mismatch. if sim(i,j) the same for data points and map points, force=0. if i and j are too far apart while they have similar sim, they are attracted. if they are too close while they have different sim, they are repelled
* plot (sim_map(i, j), strength) for all distances between the map points (i,j)
* we let the system evolve according to law of physics
* mathematically, what we're doing is that we minimize the KL divergence between the sim matrices. (formula of KL and its gradient) gradient descent. it is remarkable that this sound mathematical model corresponds to this intuitive physical formulation.
* screenshot from the paper [Hinton, Roweis, 2002]

Limitations of SNE

* First, it is not symmetric, this is solved by considering **
* More importantly,

Some imports.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
# That's an impressive list of imports!
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel

# matplotlib for graphics
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We'll generate an animation with matplotlib and moviepy
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

# Some matplotlib params.
%matplotlib inline
matplotlib.rcParams.update({'font.size': 22})
</pre>

Illustration on digit dataset.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
digits = load_digits()
tsne = TSNE()
digits_proj = tsne.fit_transform(digits.data)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def scatter(x, colors):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=colors.astype(np.int));
    plt.xlim(-25, 25);
    plt.ylim(-25, 25);
    ax.axis('off');

    txts = []
    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
scatter(digits_proj, digits.target);
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
#X = scale(X)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
distances = pairwise_distances(X, squared=True)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def _joint_probabilities_constant_sigma(D, sigma):
    P = np.exp(-D**2/2*sigma**2)
    P /= np.sum(P, axis=1)
    return P
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
D = pairwise_distances(X, squared=True)
P_constant = _joint_probabilities_constant_sigma(D, .002)
P_binary = _joint_probabilities(D, 30., False)
P_binary_s = squareform(P_binary)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(D, interpolation='none')
plt.axis('off')
plt.title("Distance matrix", fontdict={'fontsize': 16})

plt.subplot(132)
plt.imshow(P_constant, interpolation='none')
plt.axis('off')
plt.title("$p_{j|i}$ (constant sigma)", fontdict={'fontsize': 16})

plt.subplot(133)
plt.imshow(P_binary_s, interpolation='none')
plt.axis('off')
plt.title("$p_{j|i}$ (binary search sigma)", fontdict={'fontsize': 16});
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
positions = []
def _gradient_descent(objective, p0, it, n_iter, n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=[]):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        positions.append(p.copy())
        new_error, grad = objective(p, *args)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

    return p, error, i
sklearn.manifold.t_sne._gradient_descent = _gradient_descent
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tsne = TSNE()
X_proj = tsne.fit_transform(X)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
X_iter = np.dstack(position.reshape(-1, 2)
                   for position in positions)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
f, ax, sc, txts = scatter(X_iter[..., -1], y);

def make_frame_mpl(t):
    i = int(t*40)
    x = X_iter[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(10), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl,
                          duration=X_iter.shape[2]/40.)
animation.write_gif("anim.gif", fps=20)
</pre>

<img src="anim.gif" />

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
n = 1. / (pdist(X_iter[..., -1], "sqeuclidean") + 1)
Q = n / (2.0 * np.sum(n))
Q = squareform(Q)

f = plt.figure(figsize=(6, 6))
ax = plt.subplot(aspect='equal')
im = ax.imshow(Q)
plt.axis('tight');
plt.axis('off');
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def make_frame_mpl(t):
    i = int(t*40)
    n = 1. / (pdist(X_iter[..., i], "sqeuclidean") + 1)
    Q = n / (2.0 * np.sum(n))
    Q = squareform(Q)
    im.set_data(Q)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl,
                          duration=X_iter.shape[2]/40.)
animation.write_gif("anim2.gif", fps=20)
</pre>

<img src="anim2.gif" />

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([]);
    ax.get_xaxis().tick_bottom()
    ax.set_xticks([0., .5, 1.]);
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
npoints = 1000
plt.figure(figsize=(14, 3))
for i, D in enumerate((2, 5, 10)):
    u = np.random.randn(npoints, D)
    u /= norm(u, axis=1)[:, None]
    r = np.random.rand(npoints, 1)
    points = u * r**(1./D)
    ax = plt.subplot(1, 3, i+1)
    ax.set_xlabel('Ball radius')
    if i == 0:
        ax.set_ylabel('Distance from\norigin')
    simpleaxis(ax)
    ax.hist(norm(points, axis=1), 
            bins=np.linspace(0., 1., 50))
    ax.set_title('D=%d' % D, loc='left')
</pre>

Equations:

<span class="math-tex" data-type="tex">\\(p_{j|i} = \frac{\exp(-\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\lVert\mathbf{x}_i - \mathbf{x}_k\rVert^2 / 2\sigma_i^2)}\\)</span>

<span class="math-tex" data-type="tex">\\(p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}\\)</span>

<span class="math-tex" data-type="tex">\\(q_{ij} = \frac{(1 + \lVert \mathbf{y}_i - \mathbf{y}_j\rVert^2)^{-1}}{\sum_{k \neq l} (1 + \lVert \mathbf{y}_k - \mathbf{y}_l\rVert^2)^{-1}}\\)</span>

<span class="math-tex" data-type="tex">\\(KL(P||Q) = \sum_{i \neq j} p_{ij} \, \log \frac{p_{ij}}{q_{ij}}\\)</span>

Links:

* [Original paper](http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
* [Optimized t-SNE paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)
* [A notebook on t-SNE by Alexander Flabish](http://nbviewer.ipython.org/urls/gist.githubusercontent.com/AlexanderFabisch/1a0c648de22eff4a2a3e/raw/59d5bc5ed8f8bfd9ff1f7faa749d1b095aa97d5a/t-SNE.ipynb)
* [Official t-SNE page](http://lvdmaaten.github.io/tsne/)
* [scikit documentation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
* [Barnes-Hut t-SNE implementation in Python](https://github.com/danielfrg/tsne)
* [Barnes-Hut on Wikipedia](http://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
* [t-SNE on Wikipedia](http://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
* [Implementation in scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/t_sne.py)
