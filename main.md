# An illustrated introduction to t-SNE

Some imports.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy as np
from numpy import linalg
from numpy.linalg import norm
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
%matplotlib inline
</pre>

Illustration on digit dataset.

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
digits = load_digits()
tsne = TSNE()
Y = tsne.fit_transform(digits.data)
plt.figure(figsize=(6, 6))
ax = plt.subplot(aspect='equal')
ax.scatter(Y[:,0], Y[:,1], lw=0, s=40,
            c=digits.target);
for i in range(10):
    xtext, ytext = Y[digits.target == i, :].mean(axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
ax.axis('tight');
ax.axis('off');
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
n_samples = 1000
n_dims = 3
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def random_points_sphere(n_samples, n_dims, radius=1., width=0.):
    x = np.random.normal(size=(n_samples, n_dims))
    widths = width * np.random.uniform(size=(n_samples, 1), 
                                       low=-width // 2, 
                                       high=width // 2)
    a = (radius + widths) / norm(x, axis=1)[:, None]
    x *= a
    return x
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
x = np.empty((n_samples, n_dims))
x[:n_samples // 2,:] = random_points_sphere(n_samples // 2, n_dims,
                                            radius=1., width=.25)
x[n_samples // 2:,:] = random_points_sphere(n_samples // 2, n_dims,
                                            radius=2., width=.25)
clusters = np.hstack((np.zeros(n_samples // 2, dtype=np.int),
                      np.ones(n_samples // 2, dtype=np.int)))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
colors = np.array([55,126,184,
                    228,26,28]).reshape((2, 3))/255.
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
def scatter(x):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:,0], x[:,1], lw=0, s=40,
               c=colors[clusters]);
    ax.axis('tight');
    ax.axis('off');
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
scatter(x)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
scatter(PCA().fit_transform(x))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tsne = TSNE()
y = tsne.fit_transform(x)
scatter(y)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
distances = pairwise_distances(x, squared=True)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
distances.shape
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
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
D = pairwise_distances(x, squared=True)
P_constant = _joint_probabilities_constant_sigma(D, 5.)
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
    
    positions.append(p)
    return p, error, i
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
sklearn.manifold.t_sne._gradient_descent = _gradient_descent
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tsne = TSNE()
y = tsne.fit_transform(x)
scatter(y)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">

</pre>

Equations:

```
<span class="math-tex" data-type="tex">$$p_{j|i} = \frac{\exp(-\lVert\mathbf{x}_i - \mathbf{x}_j\rVert^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\lVert\mathbf{x}_i - \mathbf{x}_k\rVert^2 / 2\sigma_i^2)}$$</span>

<span class="math-tex" data-type="tex">$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$</span>

<span class="math-tex" data-type="tex">$$q_{ij} = \frac{(1 + \lVert \mathbf{y}_i - \mathbf{y}_j\rVert^2)^{-1}}{\sum_{k \neq l} (1 + \lVert \mathbf{y}_k - \mathbf{y}_l\rVert^2)^{-1}}$$</span>

<span class="math-tex" data-type="tex">$$KL(P||Q) = \sum_{i \neq j} p_{ij} \, \log \frac{p_{ij}}{q_{ij}}$$</span>
```

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
