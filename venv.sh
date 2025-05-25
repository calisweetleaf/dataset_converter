#!/bin/sh
# Virtual Environment Setup Script for Dataset Generator Original
# This script sets up a complete Python virtual environment for the dataset converter
# Compatible with POSIX sh, bash, dash, zsh, etc.

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color codes for output (may not work in all shells)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    printf "%b[%s]%b %s\n" "$BLUE" "$(date +'%Y-%m-%d %H:%M:%S')" "$NC" "$1"
}

success() {
    printf "%b[SUCCESS]%b %s\n" "$GREEN" "$NC" "$1"
}

warning() {
    printf "%b[WARNING]%b %s\n" "$YELLOW" "$NC" "$1"
}

error() {
    printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Python version
get_python_version() {
    python3 --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' || echo "0.0"
}

# Function to compare version numbers (returns 0 if $1 >= $2)
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]
}

log "Starting Dataset Generator Original Virtual Environment Setup"
log "=================================================="

# Check if we're in the correct directory
if [ ! -f "dataset_generator_original.py" ]; then
    error "dataset_generator_original.py not found. Please run this script from the dataset_converter directory."
    exit 1
fi

# Detect operating system (portable)
OS="Unknown"
UNAME_OUT="$(uname -s)"
case "$UNAME_OUT" in
    Linux*)     OS="Linux" ;;
    Darwin*)    OS="macOS" ;;
    CYGWIN*)    OS="Windows" ;;
    MINGW*)     OS="Windows" ;;
    MSYS*)      OS="Windows" ;;
    *)          OS="Unknown" ;;
esac
log "Detected OS: $OS"

# Check for Python 3
PYTHON_CMD=""
PYTHON_VERSION=""
if command_exists python3; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(get_python_version)
elif command_exists python; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
else
    error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

if version_ge "$PYTHON_VERSION" "3.8"; then
    log "Found Python $PYTHON_VERSION"
else
    error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check for pip
PIP_CMD=""
if command_exists pip3; then
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_CMD="pip"
else
    error "pip not found. Please install pip."
    exit 1
fi
log "Using pip command: $PIP_CMD"

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    warning "Virtual environment directory '$VENV_DIR' already exists."
    printf "Do you want to remove it and create a fresh one? (y/N): "
    read REPLY
    if [ "x$REPLY" = "xy" ] || [ "x$REPLY" = "xY" ]; then
        log "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        log "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    success "Virtual environment created successfully"
fi

# Activate virtual environment (portable)
log "Activating virtual environment..."
if [ "$OS" = "Windows" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
fi
if [ -f "$ACTIVATE_SCRIPT" ]; then
    . "$ACTIVATE_SCRIPT"
else
    error "Activation script not found: $ACTIVATE_SCRIPT"
    exit 1
fi

# Verify activation
if [ "x${VIRTUAL_ENV:-}" != "x" ]; then
    success "Virtual environment activated: $VIRTUAL_ENV"
else
    error "Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip, setuptools, and wheel
log "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    log "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    warning "requirements.txt not found. Installing core dependencies manually..."
    log "Installing core GUI and data processing dependencies..."
    pip install "PyQt5>=5.15.0" "pandas>=1.3.0" "numpy>=1.21.0" "rich>=10.0.0" "tqdm>=4.60.0"
    log "Installing optional dependencies..."
    pip install "pyarrow>=5.0.0" "h5py>=3.3.0" "PyYAML>=5.4.0" "lxml>=4.6.0" "fastavro>=1.4.0" "openpyxl>=3.0.0"
    pip install "scikit-learn>=1.0.0" "Faker>=8.0.0" "matplotlib>=3.4.0" "seaborn>=0.11.0"
    log "Installing and setting up NLTK..."
    pip install "nltk>=3.6.0"
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"
fi

# Create required directories
log "Creating required directories..."
mkdir -p dataset_input dataset_output dataset_logs

# Run validation tests
log "Running validation tests..."

# Test Python syntax
log "Testing Python syntax..."
python -m py_compile dataset_generator_original.py
if [ $? -eq 0 ]; then
    success "Python syntax validation passed"
else
    error "Python syntax validation failed"
fi

# Test imports
log "Testing critical imports..."
python -c "
import sys
failed_imports = []
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    print('✓ PyQt5 import successful')
except ImportError as e:
    failed_imports.append(f'PyQt5: {e}')
try:
    import pandas as pd
    print('✓ pandas import successful')
except ImportError as e:
    failed_imports.append(f'pandas: {e}')
try:
    import numpy as np
    print('✓ numpy import successful')
except ImportError as e:
    failed_imports.append(f'numpy: {e}')
try:
    from rich.console import Console
    print('✓ rich import successful')
except ImportError as e:
    failed_imports.append(f'rich: {e}')
if failed_imports:
    print('Failed imports:')
    for imp in failed_imports:
        print(f'  ✗ {imp}')
    sys.exit(1)
else:
    print('✓ All critical imports successful')
"
if [ $? -eq 0 ]; then
    success "Import tests passed"
else
    error "Import tests failed"
    exit 1
fi

# Run PowerShell scripts if available (Windows/WSL)
if command_exists pwsh || command_exists powershell; then
    log "Running PowerShell validation scripts..."
    if [ -f "file_finalizer_check.ps1" ]; then
        log "Running file finalizer check..."
        if command_exists pwsh; then
            pwsh -File file_finalizer_check.ps1 -Path . -Recurse
        else
            powershell -File file_finalizer_check.ps1 -Path . -Recurse
        fi
    fi
    if [ -f "universaL_syntax_checker.ps1" ]; then
        log "Running syntax checker..."
        if command_exists pwsh; then
            pwsh -File universaL_syntax_checker.ps1 -Path . -Recurse
        else
            powershell -File universaL_syntax_checker.ps1 -Path . -Recurse
        fi
    fi
else
    warning "PowerShell not available. Skipping PowerShell validation scripts."
fi

# Create activation script for convenience
log "Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/sh
# Activation script for Dataset Generator Original
if [ -d "venv" ]; then
    UNAME_OUT="$(uname -s)"
    case "$UNAME_OUT" in
        CYGWIN*|MINGW*|MSYS*)
            . venv/Scripts/activate
            ;;
        *)
            . venv/bin/activate
            ;;
    esac
    echo "Virtual environment activated. You can now run:"
    echo "  python dataset_generator_original.py"
    echo "  python dataset_generator_original.py --help"
else
    echo "Virtual environment not found. Please run ./venv.sh first."
fi
EOF
chmod +x activate_env.sh

# Create deactivation script
cat > deactivate_env.sh << 'EOF'
#!/bin/sh
# Deactivation script for Dataset Generator Original
if [ "x${VIRTUAL_ENV:-}" != "x" ]; then
    deactivate
    echo "Virtual environment deactivated."
else
    echo "No virtual environment is currently active."
fi
EOF
chmod +x deactivate_env.sh

# Display summary
log "=================================================="
success "Virtual Environment Setup Complete!"
log "=================================================="
echo
printf "Summary:\n"
echo "  ✓ Python $PYTHON_VERSION virtual environment created"
echo "  ✓ Required dependencies installed"
echo "  ✓ Required directories created"
echo "  ✓ Validation tests passed"
echo
printf "Next steps:\n"
echo "  1. The virtual environment is currently active"
echo "  2. Run: python dataset_generator_original.py"
echo "  3. To activate later: . activate_env.sh"
echo "  4. To deactivate: . deactivate_env.sh"
echo
printf "Available commands:\n"
echo "  python dataset_generator_original.py                    # Start GUI mode"
echo "  python dataset_generator_original.py --help             # Show help"
echo "  python dataset_generator_original.py --mode cli         # CLI mode"
echo
printf "Directories created:\n"
echo "  dataset_input/   - Place your input files here"
echo "  dataset_output/  - Generated files will be saved here"
echo "  dataset_logs/    - Application logs"
echo
success "Setup complete! You can now use the Dataset Generator."
