import tensorflow as tf

def load_tf_model(model_path):
    # Load a (frozen) Tensorflow model into memory.
    with tf.gfile.GFile(model_path, 'rb') as fid:
        model_bytes = fid.read()

    return model_bytes


def parse_tf_model(model_bytes):
    # Parse the tensorflow graph
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_bytes)

    # Load the model as a graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        tf.import_graph_def(graph_def, name='')
        tf.import_graph_def(graph_def, name='')

    # Create a session for later use
    persistent_sess = tf.Session(graph=detection_graph)

    return detection_graph, persistent_sess