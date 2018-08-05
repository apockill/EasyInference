from setuptools import setup


scripts = ["./model_prep/freeze_tf_ckpt.py"]

setup(
    name='easyinference',
    scripts=scripts,
    version='0.1',
    description='This is a helper library for me to write simple easyinference wrappers over pretrained TF models, that '
                'require none of the original model definition code. ',
    packages=['easyinference'],
    install_requires=["tensorflow-gpu",
                      "plotly",
                      "h5py"],
    zip_safe=False
)