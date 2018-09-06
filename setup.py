from setuptools import setup, find_packages

setup(name='OCR',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='OCR Keras su Cloud ML Engine',
      author='Cristiano Casadei',
      author_email='ccasadei74@gmail.com',
      license='MIT',
      install_requires=[
          'keras', 'opencv-python', 'h5py'
      ],
      zip_safe=False)
