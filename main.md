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
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import scale
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
MACHINE_EPSILON = np.finfo(np.double).eps
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
digits_proj = tsne.fit_transform(digits.data)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
plt.figure(figsize=(6, 6))
ax = plt.subplot(aspect='equal')
ax.scatter(digits_proj[:,0], digits_proj[:,1], lw=0, s=40,
            c=digits.target);
for i in range(10):
    xtext, ytext = digits_proj[digits.target == i, :].mean(axis=0)
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
        xtext, ytext = np.median(x[y == i, :], axis=0)
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
X = np.vstack([digits.data[digits.target==i] for i in range(10)])
y = np.hstack([digits.target[digits.target==i] for i in range(10)])
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
X = scale(X)
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
positions_arr = np.dstack(position.reshape(-1, 2) 
                          for position in positions)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
f, ax, sc, txt = scatter(positions_arr[..., -1], y);
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
y
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

f, ax, sc, txts = scatter(positions_arr[..., -1], y);

def make_frame_mpl(t):
    i = int(t*40)
    x = positions_arr[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(10), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl, 
                          duration=positions_arr.shape[2]/40.)
animation.write_gif("anim.gif", fps=20)
</pre>

<img src="anim.gif" />

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
