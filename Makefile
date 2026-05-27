.PHONY: all
all: install

# Install all dependencies from the lockfile.
.PHONY: install
install:
	uv sync --locked --no-install-project

# Update dependencies and the lockfile.
.PHONY: update
update:
	uv sync --no-install-project
