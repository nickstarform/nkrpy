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

clean:
	-@for f in $$(find ./nkrpy/ -type f -name "*.pyx"); do path=$$(dirname "$${f}"); fname=$$(basename "$${f}"); fname="$${fname::-4}"; for i in "c" "h" "*so"; do f=$$(find "$${path}" -type f -name "$${fname}.$${i}"); if $$(test -e "$${f}"); then rm -f "$${f}"; fi; done; done

clear_pycache:
	-@find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf

test:
	tox

dev:
	@python3 setup.py build_ext --inplace

build: clean
	@python3 setup.py build && python3 setup.py build_ext --inplace && python3 setup.py sdist

