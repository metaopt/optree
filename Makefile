# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

PROJECT_NAME   = optree
COPYRIGHT      = "MetaOPT Team. All Rights Reserved."
SHELL          = /bin/bash
.SHELLFLAGS    := -eu -o pipefail -c
PROJECT_PATH   = $(PROJECT_NAME)
SOURCE_FOLDERS = $(PROJECT_PATH) include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.py" -o -iname "*.pyi") setup.py benchmark.py
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.[ch]pp" -o -iname "*.cc" -o -iname "*.c" -o -iname "*.h")
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
_GLIBCXX_USE_CXX11_ABI ?= 1

.PHONY: default
default: install

.PHONY: install
install:
	$(PYTHON) -m pip install -v .

.PHONY: install-editable install-e
install-editable install-e:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel
	$(PYTHON) -m pip install --upgrade pybind11 cmake
	OPTREE_CXX_WERROR="$(OPTREE_CXX_WERROR)" \
		CMAKE_CXX_STANDARD="$(CMAKE_CXX_STANDARD)" \
		_GLIBCXX_USE_CXX11_ABI="$(_GLIBCXX_USE_CXX11_ABI)" \
		$(PYTHON) -m pip install -v --no-build-isolation --editable .

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

.PHONY: pre-commit-install
pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

.PHONY: python-format-install
python-format-install:
	$(call check_pip_install,ruff)

.PHONY: ruff-install
ruff-install:
	$(call check_pip_install,ruff)

.PHONY: flake8-install
flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install,flake8-bugbear)
	$(call check_pip_install,flake8-comprehensions)
	$(call check_pip_install,flake8-docstrings)
	$(call check_pip_install,flake8-pyi)
	$(call check_pip_install,flake8-simplify)

.PHONY: pylint-install
pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])
	$(call check_pip_install,pyenchant)

.PHONY: mypy-install
mypy-install:
	$(call check_pip_install,mypy)

.PHONY: xdoctest-install
xdoctest-install:
	$(call check_pip_install,xdoctest)

.PHONY: docs-install
docs-install:
	$(call check_pip_install,doc8)
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

.PHONY: clang-format-install
clang-format-install:
	$(call check_pip_install,clang-format)

.PHONY: clang-tidy-install
clang-tidy-install:
	$(call check_pip_install,clang-tidy)

.PHONY: cpplint-install
cpplint-install:
	$(call check_pip_install,cpplint)

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
	cd tests && $(PYTHON) -X dev -W 'always' -W 'error' -c 'import $(PROJECT_NAME)' && \
	$(PYTHON) -X dev -W 'always' -W 'error' -c 'import $(PROJECT_NAME)._C; print(f"GLIBCXX_USE_CXX11_ABI={$(PROJECT_NAME)._C.GLIBCXX_USE_CXX11_ABI}")' && \
	$(PYTEST) --verbose --color=yes --durations=10 --showlocals \
		--cov="$(PROJECT_NAME)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

# Python Linters

.PHONY: pre-commit
pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit --version
	$(PYTHON) -m pre_commit run --all-files

.PHONY: python-format pyfmt ruff-format
python-format pyfmt ruff-format: python-format-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff format --check . && \
	$(PYTHON) -m ruff check --select=I .

.PHONY: ruff
ruff: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check .

.PHONY: ruff-fix
ruff-fix: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check --fix --exit-non-zero-on-fix .

.PHONY: flake8
flake8: flake8-install
	$(PYTHON) -m flake8 --version
	$(PYTHON) -m flake8 --doctests --count --show-source --statistics

.PHONY: pylint
pylint: pylint-install
	$(PYTHON) -m pylint --version
	$(PYTHON) -m pylint $(PROJECT_PATH)

.PHONY: mypy
mypy: mypy-install
	$(PYTHON) -m mypy --version
	$(PYTHON) -m mypy .

.PHONY: xdoctest doctest
xdoctest doctest: xdoctest-install
	$(PYTHON) -m xdoctest --version
	$(PYTHON) -m xdoctest --global-exec "from optree import *" $(PROJECT_NAME)

# C++ Linters

.PHONY: cmake-configure
cmake-configure: cmake-install
	cmake --version
	cmake -S . -B cmake-build-debug \
		--fresh \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_STANDARD="$(CMAKE_CXX_STANDARD)" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DPython_EXECUTABLE="$(PYTHON)" \
		-DOPTREE_CXX_WERROR="$(OPTREE_CXX_WERROR)" \
		-D_GLIBCXX_USE_CXX11_ABI="$(_GLIBCXX_USE_CXX11_ABI)"

.PHONY: cmake cmake-build
cmake cmake-build: cmake-configure
	cmake --build cmake-build-debug --parallel

.PHONY: clang-format
clang-format: clang-format-install
	clang-format --version
	clang-format --style=file --Werror -i $(CXX_FILES)

.PHONY: clang-tidy
clang-tidy: clang-tidy-install cmake-configure
	clang-tidy --version
	clang-tidy --extra-arg="-v" --fix -p=cmake-build-debug $(CXX_FILES)

.PHONY: cpplint
cpplint: cpplint-install
	$(PYTHON) -m cpplint --version
	$(PYTHON) -m cpplint $(CXX_FILES)

# Documentation

.PHONY: addlicense
addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2022-$(shell date +"%Y") \
		-ignore tests/coverage.xml -check $(SOURCE_FOLDERS)

.PHONY: docstyle
docstyle: docs-install
	make -C docs clean || true
	$(PYTHON) -m doc8 docs && make -C docs html SPHINXOPTS="-W"

.PHONY: docs
docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

.PHONY: spelling
spelling: docs-install
	make -C docs clean || true
	make -C docs spelling SPHINXOPTS="-W"

.PHONY: clean-docs
clean-docs:
	make -C docs clean || true

# Utility Functions

.PHONY: lint
lint: python-format ruff flake8 pylint mypy doctest clang-format clang-tidy cpplint addlicense docstyle spelling

.PHONY: format
format: python-format-install ruff-install clang-format-install addlicense-install
	$(PYTHON) -m ruff format $(PYTHON_FILES)
	$(PYTHON) -m ruff check --fix --exit-zero .
	$(CLANG_FORMAT) -style=file -i $(CXX_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2022-$(shell date +"%Y") \
		-ignore tests/coverage.xml $(SOURCE_FOLDERS)

.PHONY: clean-python
clean-python:
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
clean: clean-python clean-build clean-docs
