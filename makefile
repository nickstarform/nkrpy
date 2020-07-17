all: build, doc

--help: help

help:
	@echo -e '\e[0;34minstall\e[m: Installs program based on location of pip3'
	@echo -e '\e[0;34mdoc\e[m: regens the doc, requires package to be built and installed already'
	@echo -e '\e[0;34mclean\e[m: cleans out temporary builds'
	@echo -e '\e[0;34mbuild\e[m: clean and rebuild in place'

install:
	@python3 -m pip install -e .

doc:
	@python3 ./bin/outlinegen.py
	@sh ./bin/docgen.sh

clean: clear_pycache
	@rm -rf ./docs ./build ./nkrpy.egg-info ./dist
	@find . -name "test-*.log" -type f -print0 | xargs -0 rm -f
	@find . -name "*.so" -type f -print0 | xargs -0 rm -f
	@for f in $(find . -name "*.pyx" -type f); do f="${f#'./'}"; for i in 'c' 'h'; do f=$(echo "${f}" | awk -F'.' '{print $1}')".${i}"; if test -f "${f}"; then rm -f "${f}"; fi; done; done

clear_pycache:
	@find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf

test:
	tox

dev:
	@python3 setup.py build_ext --inplace

build: clean
	@python3 setup.py build && python3 setup.py build_ext --inplace && python3 setup.py sdist

