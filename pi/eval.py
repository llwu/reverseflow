
def gen_y(fwd, batch_size):
    """For a function f:X->Y, generate a dataset of valid elemets y"""
    # TODO
    return y_batch


def evaluate(fwd_model, y_batch, sess, max_iterations=10000):
    """
    Solve inverse problem by search over inputs
    Given a function f, and y_batch, find x_batch s.t. f(x_batch) = y
    """
    inputs_dict = # initialize inputs
    loss = least_squares(output, y_batch)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    print("loss", loss)

    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(max_iterations=10000):
        output = sess.run({"t":train_step,"loss":loss}, feed_dict=input_feed)
        print(output)
