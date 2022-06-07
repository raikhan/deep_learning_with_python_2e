import tensorflow as tf

# Tensors can be generated similarly to numpy
t1 = tf.ones(shape=(2, 1))
t2 = tf.random.normal(shape=(3, 1))

# However, tensors are immutable
t1[0, 0] = 0.0

# Still, we need tensors to represent weights in a neural network and they need to be
# mutable. This is where tf.Variables come in
v = tf.Variable(t2)

# Variable can be mutated with .assign method
v[0, 0].assign(0.0)


# Gradient tape reminder

input_v = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    res = tf.square(input_v)
grad = tape.gradient(res, input_v)
print(grad)


# we can compute gradients of constants, not only of tf.Variables, but those have to
# be explicitly watched by the gradient tape.
# We do not want to compute gradients of everything with respect of everything by default
# (too expensive), therefore we have to explicitly state what is to be watched. Declaring
# a tf.Variable declares a value as watched by default
input_const = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(input_const)  # explicitly watching
    res = tf.square(input_const)
grad = tape.gradient(res, input_const)
print(grad)

# TF can also compute second-order gradients
time = tf.Variable(0.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time**2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
print(acceleration)
