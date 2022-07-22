from setuptools import setup


#with open('requirements.txt') as fp:
    #reqs = [l.strip() for l in fp]

description = 'ScaleApex: a simple way to combine fairscale and apex'
setup(name='scaleapex',
      version='0.1',
      description=description,
      packages=['scaleapex'],
      #install_requires=reqs,
     )

