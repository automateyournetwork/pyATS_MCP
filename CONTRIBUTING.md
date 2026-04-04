# Contributing

## Development setup

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
3. Copy the example config files:
   ```bash
   cp .env.example .env
   cp testbed.yaml.example testbed.yaml
   ```
4. Edit `.env` and `testbed.yaml` with your device details.

## Running tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=. --cov-report=html
```

## Code style

This project uses [black](https://github.com/psf/black) for formatting and [isort](https://pycqa.github.io/isort/) for import ordering.

```bash
black .
isort .
flake8 . --max-line-length=100
```

## Submitting a pull request

1. Create a feature branch from `main`.
2. Make your changes and add tests where appropriate.
3. Ensure `pytest` and linting pass locally.
4. Open a pull request against `main` with a clear description of the change.

## Security

**Never commit credentials.** Use `%ENV{VAR_NAME}` substitution in `testbed.yaml` and keep your actual testbed file out of version control (add it to `.gitignore`).

If you discover a security vulnerability, please open a GitHub issue.
