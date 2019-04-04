from tensorflow.python.client import device_lib
import tensorflow as tf


def get_all_devices():
    """Return CPU's and GPU's on this computer"""

    gpu_opts_for_process = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_opts_for_process)
    with tf.Session(config=config):
        all_devices = device_lib.list_local_devices()
    return all_devices


def get_gpus():
    all_devices = get_all_devices()

    gpus = [dev for dev in all_devices if dev.device_type == "GPU"]
    return gpus
