# Contributing to YOLOv7-Ultimate

Thank you for your interest in contributing to YOLOv7-Ultimate! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](https://github.com/liangc001/YOLOv7-Ultimate/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce (for bugs)
   - Expected behavior
   - Environment information (OS, Python version, GPU, etc.)
   - Error messages and logs

### Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests if applicable
5. Update documentation if needed
6. Commit with clear messages
7. Push to your fork
8. Create a Pull Request

### Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular
- Comment complex logic

### Commit Messages

Use clear and descriptive commit messages:

- `feat: add feature X`
- `fix: resolve issue with Y`
- `docs: update README`
- `refactor: improve code structure`
- `test: add tests for Z`

### Testing

Before submitting a PR:

1. Test your changes locally
2. Ensure existing functionality still works
3. Run any existing tests

```bash
python test.py --weights yolov7.pt --data data/coco.yaml
```

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update relevant guides in docs/

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/YOLOv7-Ultimate.git
cd YOLOv7-Ultimate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install in development mode
pip install -r requirements.txt

# Run tests
python test.py --weights yolov7.pt
```

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Areas for Contribution

- Bug fixes
- Performance improvements
- New features (with discussion first)
- Documentation improvements
- Example notebooks
- Test coverage

## Questions?

Feel free to open an issue for discussion before starting major work.

Thank you for contributing!
