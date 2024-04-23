from setuptools import setup
import setuptools
import pip

links = []
requires = []

try:
		requirements = pip.req.parse_requirements('requirements.txt')
except:
		# new versions of pip requires a session
		requirements = pip.req.parse_requirements(
				'requirements.txt', session=pip.download.PipSession())

for item in requirements:
		# we want to handle package names and also repo urls
		if getattr(item, 'url', None):  # older pip has url
				links.append(str(item.url))
		if getattr(item, 'link', None): # newer pip has link
				links.append(str(item.link))
		if item.req:
				requires.append(str(item.req))

setup(name='celldetective',
			version='1.0.2',
			description='description',
			long_description=open('README.rst',encoding="utf8").read(),
			long_description_content_type='text/markdown',
			url='http://github.com/remyeltorro/celldetective',
			author='Rémy Torro',
			author_email='remy.torro@inserm.fr',
			license='GPL-3.0',
			packages=setuptools.find_packages(),
			zip_safe=False,
			package_data={'celldetective': ['*','scripts/*','gui/*','models/*/*/*','models/*','models/*/*','icons/*','links/*','datasets/*', 'datasets/*/*']},
			entry_points = {
				'console_scripts': [
					'celldetective = celldetective.__main__:main'],
			},
			install_requires = requires,
			dependency_links = links
			)

