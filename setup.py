from setuptools import setup

setup(name='repalette',
      version='0.1',
      description='A cluster-based colour editor',
      url='https://github.com/ghager93/repalette',
      author='Gerard Hager',
      author_email='ghager93@gmail.com',
      license='MIT',
      install_requires=['numpy',
                        'scikit-learn'],
      packages=['repalette'],
      zip_safe=False)