# Chain-of-Thought-Prompting

# Project Setup Guide

This guide will walk you through setting up and running this project locally from scratch.

## Prerequisites

- Python 3.11
- Git (for cloning the repository)

## Initial Setup

### 1. Install Python 3.11

If you haven't installed Python 3.11 yet:

- Download Python 3.11 from [python.org](https://python.org)
- During installation, make sure to check "Add Python to PATH"
- Verify installation by running:

```bash
python --version  # Should show Python 3.11.x
```

### 2. Install pip (Python Package Installer)

Pip usually comes with Python, but if you need to install it:

**On Windows:**

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

**On macOS/Linux:**

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```

Verify pip installation:

```bash
pip --version
```

### 3. Install uv

uv is a fast Python package installer. Install it using pip:

```bash
pip install uv
```

### 4. Set Up Virtual Environment

Create and activate a virtual environment using uv with Python 3.11:

**On Windows:**

```bash
uv venv -p 3.11
.\.venv\Scripts\activate
```

**On macOS/Linux:**

```bash
uv venv -p 3.11
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt indicating the virtual environment is active.

### 5. Install Project Dependencies

With the virtual environment activated, install the project requirements using uv:

```bash
uv pip install -r requirements.txt
```
