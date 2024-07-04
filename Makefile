print-%: ; @echo $* = $($*)
PROJECT_NAME   = optree
COPYRIGHT      = "MetaOPT Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -name "*.h" -o -name "*.cpp")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTESTOPTS     ?=
OPTREE_CXX_WERROR ?= ON

.PHONY: default
default: install

install:
	$(PYTHON) -m pip install -vvv .

install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel
	$(PYTHON) -m pip install --upgrade pybind11 cmake
	OPTREE_CXX_WERROR="$(OPTREE_CXX_WERROR)" $(PYTHON) -m pip install -vvv --no-build-isolation --editable .

install-e: install-editable  # alias

uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	$(PYTHON) -m build

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(1))
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(2))

pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])
	$(call check_pip_install,pyenchant)

flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install,flake8-bugbear)
	$(call check_pip_install,flake8-comprehensions)
	$(call check_pip_install,flake8-docstrings)
	$(call check_pip_install,flake8-pyi)
	$(call check_pip_install,flake8-simplify)

py-format-install:
	$(call check_pip_install,isort)
	$(call check_pip_install,black)

ruff-install:
	$(call check_pip_install,ruff)

mypy-install:
	$(call check_pip_install,mypy)

xdoctest-install:
	$(call check_pip_install,xdoctest)

pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

docs-install:
	$(call check_pip_install_extra,pydocstyle,pydocstyle[toml])
	$(call check_pip_install_extra,doc8,"doc8<1.0.0a0")  # unpin this when we drop support for Python 3.7
	if ! $(PYTHON) -c "import sys; exit(sys.version_info < (3, 8))"; then \
		$(PYTHON) -m pip uninstall --yes importlib-metadata; \
		$(call check_pip_install_extra,importlib-metadata,"importlib-metadata<5.0.0a0"); \
	fi
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-rtd-theme)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install_extra,sphinxcontrib-spelling,sphinxcontrib-spelling pyenchant)

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

test-install: pytest-install
	$(PYTHON) -m pip install --requirement tests/requirements.txt

cmake-install:
	command -v cmake || $(call check_pip_install,cmake)

cpplint-install:
	$(call check_pip_install,cpplint)

clang-format-install:
	$(call check_pip_install,clang-format)

clang-tidy-install:
	$(call check_pip_install,clang-tidy)

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang && sudo ln -sf /usr/lib/go/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

pytest: pytest-install
	$(PYTHON) -m pytest --version
	cd tests && $(PYTHON) -X dev -c 'import $(PROJECT_PATH)' && \
	$(PYTHON) -X dev -c 'import $(PROJECT_PATH)._C; print(f"GLIBCXX_USE_CXX11_ABI={$(PROJECT_PATH)._C.GLIBCXX_USE_CXX11_ABI}")' && \
	$(PYTHON) -X dev -m pytest --verbose --color=yes \
		--cov="$(PROJECT_PATH)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

test: pytest

# Python linters

pylint: pylint-install
	$(PYTHON) -m pylint --version
	$(PYTHON) -m pylint $(PROJECT_PATH)

flake8: flake8-install
	$(PYTHON) -m flake8 --version
	$(PYTHON) -m flake8 --doctests --count --show-source --statistics

py-format: py-format-install
	$(PYTHON) -m isort --version
	$(PYTHON) -m black --version
	$(PYTHON) -m isort --project $(PROJECT_PATH) --check $(PYTHON_FILES) && \
	$(PYTHON) -m black --check $(PYTHON_FILES)

ruff: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check .

ruff-fix: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check . --fix --exit-non-zero-on-fix

mypy: mypy-install
	$(PYTHON) -m mypy --version
	$(PYTHON) -m mypy $(PROJECT_PATH) --install-types --non-interactive

xdoctest: xdoctest-install
	$(PYTHON) -m xdoctest --version
	$(PYTHON) -m xdoctest --global-exec "from optree import *" $(PROJECT_PATH)

doctest: xdoctest

pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit --version
	$(PYTHON) -m pre_commit run --all-files

# C++ linters

cmake-configure: cmake-install
	cmake --version
	cmake -S . -B cmake-build-debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DPYTHON_EXECUTABLE="$(PYTHON)" \
		-DOPTREE_CXX_WERROR="$(OPTREE_CXX_WERROR)"

cmake-build: cmake-configure
	cmake --build cmake-build-debug --parallel

cmake: cmake-build

cpplint: cpplint-install
	$(PYTHON) -m cpplint --version
	$(PYTHON) -m cpplint $(CXX_FILES)

clang-format: clang-format-install
	clang-format --version
	clang-format --style=file -i $(CXX_FILES) -n --Werror

clang-tidy: clang-tidy-install cmake-configure
	clang-tidy --version
	clang-tidy --extra-arg="-v" -p=cmake-build-debug $(CXX_FILES)

# Documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") -check $(SOURCE_FOLDERS)

docstyle: docs-install
	make -C docs clean
	$(PYTHON) -m pydocstyle $(PROJECT_PATH) && doc8 docs && make -C docs html SPHINXOPTS="-W"

docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

spelling: docs-install
	make -C docs clean
	make -C docs spelling SPHINXOPTS="-W"

clean-docs:
	make -C docs clean

# Utility functions

lint: ruff flake8 py-format mypy pylint doctest clang-format clang-tidy cpplint addlicense docstyle spelling

format: py-format-install ruff-install clang-format-install addlicense-install
	$(PYTHON) -m isort --project $(PROJECT_PATH) $(PYTHON_FILES)
	$(PYTHON) -m black $(PYTHON_FILES)
	$(PYTHON) -m ruff check . --fix --exit-zero
	$(CLANG_FORMAT) -style=file -i $(CXX_FILES)
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") $(SOURCE_FOLDERS)

clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm tests/.coverage
	rm tests/coverage.xml

clean-build:
	rm -rf build/ dist/
	rm -rf *.egg-info .eggs

clean: clean-py clean-build clean-docs
