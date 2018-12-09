import tensorflow as tf


def load_tf_model(model_path):
    # Load a (frozen) Tensorflow model into memory.
    with tf.gfile.GFile(str(model_path), 'rb') as fid:
        model_bytes = fid.read()
        return model_bytes


def parse_tf_model(model_bytes):
    # Parse the tensorflow graph
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_bytes)

    # Load the model definition
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            producer_op_list=None,
                            name='')

    # Create a session for later use
    sess = tf.Session(graph=graph)

    return graph, sess
