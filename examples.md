# Examples

This provides some specific examples of how to mark things up.  

## Markdown

Here are a few examples of how to write in [markdown](http://docs.atlas.oreilly.com/ch13.html#markdownref) for Atlas.  

### Making a Python program executable

Here's how to make a program executable:

```html
<pre data-executable="true" data-code-language="python">
...
</pre>
```

### Embedding an equation


Wrap Latex equations in a `<span data-type="tex"> ... </span>`.

<span data-type="tex">$$a^2 + b^2 = c^2$$</span>

<span class="math-tex" data-type="tex">\(a^2 + b^2 = c^2\)</span>


## AsciiDoc

Here are a few examples of writing in [AsciiDoc](http://docs.atlas.oreilly.com/ch12.html#asciidocref).

### Making a program executable

Add the `[data-executable="true"]` macro to make your source listing executable, like this:

```html
[source, python]
[data-executable="true"]
----
portfolio = [
   {'name': 'IBM', 'shares': 100, 'price': 91.1},
   {'name': 'AAPL', 'shares': 50, 'price': 543.22},
   {'name': 'FB', 'shares': 200, 'price': 21.09},
   {'name': 'HPQ', 'shares': 35, 'price': 31.75},
   {'name': 'YHOO', 'shares': 45, 'price': 16.35},
   {'name': 'ACME', 'shares': 75, 'price': 115.65}
]

cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])

print(cheap)
print(expensive)

----
````

### Embedding an equation
