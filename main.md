# An illustrated introduction to t-SNE

Hello world!

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
import numpy as np
from numpy.linalg import norm
import sklearn
import matplotlib.pyplot as plt
%matplotlib inline
</pre>

Let's test an inline equation: <span class="math-tex" data-type="tex">$\pi=3.14$</span>. And now a block equation:

<span class="math-tex" data-type="tex">$$\int_0^1 f(x)dx$$</span>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
from sklearn.manifold import TSNE
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
x0 = random_points_sphere(n_samples // 2, 3,
                          radius=1., width=.25)
x1 = random_points_sphere(n_samples // 2, 3,
                          radius=2., width=.25)
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
ax = plt.subplot(aspect='equal')
ax.plot(x0[:,0], x0[:,1], '.');
ax.plot(x1[:,0], x1[:,1], '.');
ax.axis('off');
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
tsne = TSNE()
y = tsne.fit_transform(np.vstack((x0, x1)))
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
ax = plt.subplot(aspect='equal')
ax.scatter(y[:,0], y[:,1], lw=0,
            c=np.hstack((np.zeros(n_samples // 2),
                       np.ones(n_samples // 2))));
ax.axis('off');
</pre>

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">

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
