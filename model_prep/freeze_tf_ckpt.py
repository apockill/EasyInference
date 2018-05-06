import os
import argparse
from pathlib import Path

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def save_graph_summary(session, log_dir):
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(session.graph)


def freeze_graph(model_path, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_path: The path to the basename of the model.
        Example: model_files/model where inside model_files there is
            model.data-00000-of-000001
            model.index
            model.meta
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """

    # We precise the file fullname of our freezed graph
    absolute_model_dir = Path(model_path).parent
    output_graph_path = absolute_model_dir / "frozen_model.pb"
    meta_file = list(Path(absolute_model_dir).glob("*.meta"))[0]

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(str(meta_file), clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, model_path)

        # Save a graph summary for easy tensorboard viewing
        save_graph_summary(sess, str(Path(model_path).parent))

        # Print information about the loaded model
        ops = [o.name for o in tf.get_default_graph().get_operations()]
        print("All Operations: \n", ops)

        if not output_node_names:
            print("You need to supply the name of a node to --output_node_names.")
            return -1

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(str(output_graph_path), "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_path, args.output_node_names)
