# Development Notes

## Source Layout

```text
hmod/
  Cargo.toml
  pyproject.toml
  binder/             Binder environment for running notebooks from GitHub
  src/                 Rust extension sources
  python/hmod/         Python package modules
  python/tests/        Unit and integration tests
  notebooks/           Runnable examples
  docs/                GitHub-rendered documentation
```

The package name on PyPI is `hmodFFT`, while the import name is `hmod`.

## Install From Source

From the `hmod/` directory:

```bash
pip install -e ".[tests]"
```

This uses `maturin` through `pyproject.toml` to build the Rust extension.

## Run Tests

Run the Python tests with:

```bash
pytest python/tests
```

Some tests use optional packages such as `ngsolve`, and the reference-data tests
compare the FFT implementation against dense reference matrices.

## Binder

The Binder setup in `binder/` installs `hmodFFT` from PyPI and opens the
notebooks in JupyterLab. This keeps the Binder build small because it can use
the published package instead of building the Rust extension from the repository
checkout.

When the notebook uses new API changes, publish a matching PyPI release before
expecting the Binder badge to run successfully.

## Documentation Style

The documentation is intentionally kept as plain Markdown so that GitHub renders
it without an additional documentation build step.

Recommended conventions:

- Keep `README.md` as the landing page and installation quick start.
- Put conceptual material in `docs/mathematical-background.md`.
- Put routine names, argument conventions, and small code snippets in
  `docs/matrix-assembly.md`.
- Keep notebooks as runnable examples, but mirror their core workflow in
  Markdown so users can read it quickly on GitHub.
- When a public routine changes, update both its docstring and the relevant
  `docs/` page.

If the documentation grows beyond these pages, the same Markdown files can be
used as input for a static documentation site with MkDocs or Sphinx.
