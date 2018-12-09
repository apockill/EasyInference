from setuptools import setup, find_namespace_packages


scripts = []


setup(
    name='easy_inference',
    scripts=scripts,
    version='0.5',
    description='This is a helper library for me to write simple easy_inference'
                ' wrappers over pretrained TF models, that '
                'require none of the original model definition code. ',
    packages=find_namespace_packages(),
    install_requires=["tensorflow-gpu",
                      "plotly",
                      "h5py",
                      "ujson"],
    include_package_data=True,
    zip_safe=False
)