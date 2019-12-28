all: build, doc

--help: help

help:
	@echo -e '\e[0;34minstall\e[m: Installs program based on location of pip3'
	@echo -e '\e[0;34mdoc\e[m: regens the doc, requires package to be built and installed already'
	@echo -e '\e[0;34mclean\e[m: cleans out temporary builds'
	@echo -e '\e[0;34mbuild\e[m: clean and rebuild in place'

install:
	@pip3 install -e .

doc:
	@python ./bin/outlinegen.py
	@sh ./bin/docgen.sh

clean: clear_pycache
	@rm -rf ./docs ./build ./nkrpy.egg-info

clear_pycache:
	@find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf

test:
	tox

build: clean
	@python3 setup.py build && python3 setup.py sdist

