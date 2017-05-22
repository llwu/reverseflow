[![Build Status](https://travis-ci.org/wacabanga/reverseflow.svg?branch=rewrite)](https://travis-ci.org/wacabanga/reverseflow)

# ReverseFlow

ReverseFlow is a Python3 library to execute TensorFlow programs backwards

## Depdendencies

- TensorFlow

## Quick Start

There are two things to do.
First, construct a function using a `tf.placeholder` for each input.

```python
from reverseflow import invert

## f(x,y) = x * y + x
tf.reset_default_graph()
g = tf.get_default_graph()

x = tf.placeholder('float32', name="x")
y = tf.placeholder('float32', name="y")
```

Then call `invert`

```python
inv_g = invert(g, (x, y))
```
