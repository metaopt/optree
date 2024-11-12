PROJECT_NAME   = optree
COPYRIGHT      = "MetaOPT Team. All Rights Reserved."
SHELL          = /bin/bash
.SHELLFLAGS    := -eu -o pipefail -c
PROJECT_PATH   = $(PROJECT_NAME)
SOURCE_FOLDERS = $(PROJECT_PATH) include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -name "*.h" -o -name "*.cpp")
COMMIT_HASH    = $(shell git rev-parse HEAD)
COMMIT_HASH_SHORT = $(shell git rev-parse --short=7 HEAD)
GOPATH         ?= $(HOME)/go
GOBIN          ?= $(GOPATH)/bin
PATH           := $(PATH):$(GOBIN)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTEST         ?= $(PYTHON) -X dev -m pytest
PYTESTOPTS     ?=
CMAKE_CXX_STANDARD ?= 20
OPTREE_CXX_WERROR  ?= ON

.PHONY: default
default: install

.PHONY: install
install:
	$(PYTHON) -m pip install -vv .

.PHONY: install-editable install-e
install-editable install-e:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel
	$(PYTHON) -m pip install --upgrade pybind11 cmake
	OPTREE_CXX_WERROR="$(OPTREE_CXX_WERROR)" CMAKE_CXX_STANDARD="$(CMAKE_CXX_STANDARD)" \
		$(PYTHON) -m pip install -vv --no-build-isolation --editable .

.PHONY: uninstall
uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

.PHONY: build
build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	find $(PROJECT_PATH) -type f -name '*.so' -delete
	find $(PROJECT_PATH) -type f -name '*.pxd' -delete
	rm -rf *.egg-info .eggs
	$(PYTHON) -m build --verbose

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(1))
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(2))

.PHONY: pylint-install
pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])
	$(call check_pip_install,pyenchant)

.PHONY: flake8-install
flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install,flake8-bugbear)
	$(call check_pip_install,flake8-comprehensions)
	$(call check_pip_install,flake8-docstrings)
	$(call check_pip_install,flake8-pyi)
	$(call check_pip_install,flake8-simplify)

.PHONY: py-format-install
py-format-install:
	$(call check_pip_install,isort)
	$(call check_pip_install,black)

.PHONY: ruff-install
ruff-install:
	$(call check_pip_install,ruff)

.PHONY: mypy-install
mypy-install:
	$(call check_pip_install,mypy)

.PHONY: xdoctest-install
xdoctest-install:
	$(call check_pip_install,xdoctest)

.PHONY: pre-commit-install
pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

.PHONY: docs-install
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

.PHONY: pytest-install
pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

.PHONY: test-install
test-install: pytest-install
	$(PYTHON) -m pip install --requirement tests/requirements.txt

.PHONY: cmake-install
cmake-install:
	command -v cmake || $(call check_pip_install,cmake)

.PHONY: cpplint-install
cpplint-install:
	$(call check_pip_install,cpplint)

.PHONY: clang-format-install
clang-format-install:
	$(call check_pip_install,clang-format)

.PHONY: clang-tidy-install
clang-tidy-install:
	$(call check_pip_install,clang-tidy)

.PHONY: go-install
go-install:
	command -v go || sudo apt-get satisfy -y 'golang (>= 1.16)'

.PHONY: addlicense-install
addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

.PHONY: pytest test
pytest test: pytest-install
	$(PYTEST) --version
	cd tests && $(PYTHON) -X dev -W 'always' -W 'error' -c 'import $(PROJECT_PATH)' && \
	$(PYTHON) -X dev -W 'always' -W 'error' -c 'import $(PROJECT_PATH)._C; print(f"GLIBCXX_USE_CXX11_ABI={$(PROJECT_PATH)._C.GLIBCXX_USE_CXX11_ABI}")' && \
	$(PYTEST) --verbose --color=yes --durations=10 --showlocals \
		--cov="$(PROJECT_PATH)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

# Python linters

.PHONY: pylint
pylint: pylint-install
	$(PYTHON) -m pylint --version
	$(PYTHON) -m pylint $(PROJECT_PATH)

.PHONY: flake8
flake8: flake8-install
	$(PYTHON) -m flake8 --version
	$(PYTHON) -m flake8 --doctests --count --show-source --statistics

.PHONY: py-format
py-format: py-format-install
	$(PYTHON) -m isort --version
	$(PYTHON) -m black --version
	$(PYTHON) -m isort --project $(PROJECT_PATH) --check $(PYTHON_FILES) && \
	$(PYTHON) -m black --check $(PYTHON_FILES)

.PHONY: ruff
ruff: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check .

.PHONY: ruff-fix
ruff-fix: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check --fix --exit-non-zero-on-fix .

.PHONY: mypy
mypy: mypy-install
	$(PYTHON) -m mypy --version
	$(PYTHON) -m mypy $(PROJECT_PATH)

.PHONY: xdoctest
xdoctest: xdoctest-install
	$(PYTHON) -m xdoctest --version
	$(PYTHON) -m xdoctest --global-exec "from optree import *" $(PROJECT_PATH)

.PHONY: doctest
doctest: xdoctest

.PHONY: pre-commit
pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit --version
	$(PYTHON) -m pre_commit run --all-files

# C++ linters

.PHONY: cmake-configure
cmake-configure: cmake-install
	cmake --version
	cmake -S . -B cmake-build-debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_STANDARD="$(CMAKE_CXX_STANDARD)" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DPython_EXECUTABLE="$(PYTHON)" \
		-DOPTREE_CXX_WERROR="$(OPTREE_CXX_WERROR)"

.PHONY: cmake cmake-build
cmake cmake-build: cmake-configure
	cmake --build cmake-build-debug --parallel

.PHONY: cpplint
cpplint: cpplint-install
	$(PYTHON) -m cpplint --version
	$(PYTHON) -m cpplint $(CXX_FILES)

.PHONY: clang-format
clang-format: clang-format-install
	clang-format --version
	clang-format --style=file --Werror -i $(CXX_FILES)

.PHONY: clang-tidy
clang-tidy: clang-tidy-install cmake-configure
	clang-tidy --version
	clang-tidy --extra-arg="-v" --extra-arg="-std=c++17" --fix -p=cmake-build-debug $(CXX_FILES)

# Documentation

.PHONY: addlicense
addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2022-$(shell date +"%Y") \
		-ignore tests/coverage.xml -check $(SOURCE_FOLDERS)

.PHONY: docstyle
docstyle: docs-install
	make -C docs clean
	$(PYTHON) -m pydocstyle $(PROJECT_PATH) && doc8 docs && make -C docs html SPHINXOPTS="-W"

.PHONY: docs
docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

.PHONY: spelling
spelling: docs-install
	make -C docs clean
	make -C docs spelling SPHINXOPTS="-W"

.PHONY: clean-docs
clean-docs:
	make -C docs clean

# Utility functions

.PHONY: lint
lint: ruff flake8 py-format mypy pylint doctest clang-format clang-tidy cpplint addlicense docstyle spelling

.PHONY: format
format: py-format-install ruff-install clang-format-install addlicense-install
	$(PYTHON) -m isort --project $(PROJECT_PATH) $(PYTHON_FILES)
	$(PYTHON) -m black $(PYTHON_FILES)
	$(PYTHON) -m ruff check --fix --exit-zero .
	$(CLANG_FORMAT) -style=file -i $(CXX_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2022-$(shell date +"%Y") \
		-ignore tests/coverage.xml $(SOURCE_FOLDERS)

.PHONY: clean-py
clean-py:
	find . -type f -name '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm -f tests/.coverage tests/.coverage.* tests/coverage.xml tests/coverage-*.xml tests/coverage.*.xml
	rm -f tests/.junit tests/.junit.* tests/junit.xml tests/junit-*.xml tests/junit.*.xml

.PHONY: clean-build
clean-build:
	rm -rf build/ dist/ cmake-build/ cmake-build-*/
	find $(PROJECT_PATH) -type f -name '*.so' -delete
	find $(PROJECT_PATH) -type f -name '*.pxd' -delete
	rm -rf *.egg-info .eggs

.PHONY: clean
clean: clean-py clean-build clean-docs
