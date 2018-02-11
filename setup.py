from setuptools import setup


scripts = ["./model_prep/freeze_tf_ckpt.py"]

setup(
    name='inference',
    scripts=scripts,
    version='0.1',
    description='This is a helper library for me to write simple inference wrappers over pretrained TF models, that '
                'require none of the original model definition code. ',
    packages=['inference'],
    install_requires=[],
    zip_safe=False
)