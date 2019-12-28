from setuptools import setup, find_packages
import os


def populate_info(kwargs):
    with open('nkrpy/__info__.py', 'w') as f:
        for key, val in kwargs.items():
            if isinstance(val, str):
                f.write(f'{key} = "{val}"\n')
            if isinstance(val, bool) or\
               isinstance(val, int) or\
               isinstance(val, float):
                f.write(f'{key} = {val}\n')
            if isinstance(val, list):
                f.write(f'{key} = {val}\n')


with open(os.path.join(os.path.dirname(__file__), '.version')) as v:
    version = v.read()

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as r:
    skiprows = 2
    while True:
        for i in range(skiprows):
            readme = (r.readline()).strip('\n').strip(' ').strip('-')
        if readme != '':
            break

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as req:
    requirements = req.read().splitlines()

package_data = {
    '': ['LICENSE', 'README.md', 'requirements.txt', '.version']
}

settings = {
    'name': 'nkrpy',
    'description': readme,
    'long_description': readme,
    'long_description_content_type': "text/md",
    'version': version,
    'url': 'http://github.com/nickalaskreynolds/nkrpy',
    'author': 'Nickalas Reynolds',
    'author_email': 'email@nickreynolds.xyz',
    'author_website': 'http://nickreynolds.xyz',
    'license': 'MPL2.0',
    'packages': find_packages(exclude=['tests', 'docs', 'examples']),
    'scripts': [],
    'zip_safe': False,
    'include_package_data': True,
    'package_data': package_data,
    'install_requires': requirements,
    'classifiers': [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Intended Audience :: Developers',
        'Natural Language :: English'
    ]
}

if __name__ == '__main__':
    populate_info(settings)
    setup(**settings)

# end of file
