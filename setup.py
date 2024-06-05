from setuptools import setup
import setuptools
import pip
import os

from pip._internal.req import parse_requirements

links = []
requires = []

requirements = parse_requirements('requirements.txt', session='hack')
requirements = list(requirements) 
try:
    requirements = [str(ir.req) for ir in requirements]
except:
    requirements = [str(ir.requirement) for ir in requirements]


# for item in requirements:
# 		# we want to handle package names and also repo urls
# 		if getattr(item, 'url', None):  # older pip has url
# 				links.append(str(item.url))
# 		if getattr(item, 'link', None): # newer pip has link
# 				links.append(str(item.link))
# 		if item.req:
# 				requires.append(str(item.req))

setup(name='celldetective',
			version='1.1.0',
			description='description',
			long_description=open('README.rst',encoding="utf8").read(),
			long_description_content_type='text/markdown',
			url='http://github.com/remyeltorro/celldetective',
			author='Rémy Torro',
			author_email='remy.torro@inserm.fr',
			license='GPL-3.0',
			packages=setuptools.find_packages(),
			zip_safe=False,
			package_data={'celldetective': ['*',os.sep.join(['scripts','*']),os.sep.join(['gui','*']),os.sep.join(['models','*','*','*']),os.sep.join(['models','*']),os.sep.join(['models','*','*']),os.sep.join(['icons','*']),os.sep.join(['links','*']),os.sep.join(['datasets','*']), os.sep.join(['datasets','*','*'])]},
			entry_points = {
				'console_scripts': [
					'celldetective = celldetective.__main__:main'],
			},
			install_requires = requirements,
			#dependency_links = links
			)

