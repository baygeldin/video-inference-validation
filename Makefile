.PHONY: all
all: install

# Install all dependencies from the lockfile.
.PHONY: install
install:
	uv sync --locked

# Update dependencies and the lockfile.
.PHONY: update
update:
	uv sync
