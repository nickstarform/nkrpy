all: doc, clean, build

install:
	@pip3 install -e .

doc:
	@python ./bin/outlinegen.py
	@sh ./bin/outlinegen.sh
	@sh ./bin/docgen.sh

clean: clear_pycache
	@rm -rf ./docs ./build ./ nkrpy.egg-info

clear_pycache:
	@find . -name "__pycache__" -type d -exec '{rm -rf}'

build:
	@python3 setup.py build && python3 setup.py sdist

