# Writing in Markdown

Here are a few examples of how to write in [markdown](http://docs.atlas.oreilly.com/ch13.html#markdownref) for Atlas.  

## Making a Python program executable

Here's how to make a program executable:

```html
<pre data-executable="true" data-code-language="python">
...
</pre>
```

Here's an example:

<pre data-executable="true" data-code-language="python">
%pylab inline
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));
</pre>

## Embedding an equation

Wrap Latex equations in a `<span data-type="tex">$$ your latex equation here $$</span>`.  For example:

<span data-type="tex">$$a^2 + b^2 = c^2$$</span>
