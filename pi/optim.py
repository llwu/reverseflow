import tensorflow as tf

def accumulate_mean_error(errors):
    return tf.add_n(errors)/len(errors)

def minimize_error(inv_g, input_feed, sess):
    # params = inv_g.get_collection("params")
    errors = inv_g.get_collection("errors")
    loss = accumulate_mean_error(errors)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    print("loss", loss)

    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(10000):
        output = sess.run({"t":train_step,"loss":loss}, feed_dict=input_feed)
        print(output)
