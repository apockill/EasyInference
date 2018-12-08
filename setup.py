from setuptools import setup


scripts = []


setup(
    name='easy_inference',
    scripts=scripts,
    version='0.1',
    description='This is a helper library for me to write simple easy_inference'
                ' wrappers over pretrained TF models, that '
                'require none of the original model definition code. ',
    packages=['easy_inference'],
    install_requires=["tensorflow-gpu",
                      "plotly",
                      "h5py",
                      "ujson"],
    zip_safe=False
)