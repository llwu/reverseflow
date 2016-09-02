## Parametric Inversion

Parametric Inversion is a library for inverting (possibly non-invertible functions).
It is built on top of tensorflow in python.

## Quick Start

There are two things to do.
First, construct a function using a `tf.placeholder` for each input.

python
```
import pi
from pi import invert
import tensorflow as tf
from tensorflow import float32

## f(x,y) = x * y + x
tf.reset_default_graph()
g = tf.get_default_graph()

x = tf.placeholder(float32, name="x")
y = tf.placeholder(float32, name="y")
```

Then call `pi.invert`

python
```
inv_g = pi.invert.invert(g, (x, y))
```
