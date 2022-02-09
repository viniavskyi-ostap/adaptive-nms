from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Adaptive keypoints NMS by ',
    ext_modules=cythonize("ssc.pyx", language_level=3),
    zip_safe=False,
)
