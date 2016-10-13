## reverseflow

reverseflow is a library to execute tensorflow programs backwards.

## Install

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

Then call `rf.invert`

python
```
inv_g = rf.invert.invert(g, (x, y))
```
