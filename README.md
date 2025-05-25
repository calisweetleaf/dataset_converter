# Dataset Generator Original - Advanced LLM Fine-Tuning Dataset Toolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Overview

A comprehensive GUI toolkit for generating, formatting, and processing datasets specifically designed for fine-tuning language models. This application provides an all-in-one solution for dataset preparation with support for multiple formats, data validation, augmentation, and visualization.

### ✨ Key Features

- **🔄 Format Conversion**: Support for 12+ data formats (JSON, JSONL, CSV, Parquet, HDF5, XML, YAML, Avro, Excel, SQLite, TSV)
- **🎯 Template-Based Generation**: Pre-configured templates for instruction-following, chat, and Q&A datasets
- **🔧 Auto-Fix & Validation**: Intelligent data cleaning and validation with detailed reporting
- **📊 Data Visualization**: Interactive charts and statistical analysis
- **🎭 Data Augmentation**: Text rephrasing, synonym replacement, and data expansion
- **🔒 Data Anonymization**: Faker-based sensitive data anonymization
- **✂️ Dataset Splitting**: Train/validation/test set creation with customizable ratios
- **🖥️ Modern GUI**: Dark theme, drag-and-drop support, tabbed interface
- **🔍 Advanced Validation**: Custom validation rules, type checking, duplicate detection
- **📈 Statistics & Analytics**: Comprehensive dataset statistics and insights

## 📋 Requirements

### Core Dependencies

```
Python 3.8+
PyQt5 >= 5.15.0
pandas >= 1.3.0
numpy >= 1.21.0
rich >= 10.0.0
tqdm >= 4.60.0
```

### Optional Dependencies (Feature-Specific)

```
# Data Format Support
pyarrow >= 5.0.0          # Parquet format
h5py >= 3.3.0             # HDF5 format  
PyYAML >= 5.4.0           # YAML format
lxml >= 4.6.0             # Enhanced XML support
fastavro >= 1.4.0         # Avro format
openpyxl >= 3.0.0         # Excel format

# Data Processing & Analysis
scikit-learn >= 1.0.0     # Dataset splitting
Faker >= 8.0.0            # Data generation & anonymization
nltk >= 3.6.0             # Text processing & augmentation

# Visualization
matplotlib >= 3.4.0       # Plotting and charts
seaborn >= 0.11.0         # Statistical visualizations
```

## 🛠️ Installation

### Method 1: Direct Installation

```bash
# Clone the repository
git clone <repository-url>
cd dataset_converter

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python dataset_generator_original.py
```

### Method 2: Using Setup Script

```bash
# Make setup script executable (Linux/Mac)
chmod +x venv.sh

# Run setup script
./venv.sh

# Or on Windows with Git Bash:
bash venv.sh
```

## 🎮 Usage

### GUI Mode (Recommended)

```bash
python dataset_generator_original.py
```

### Command Line Mode

```bash
python dataset_generator_original.py --mode cli --input data.json --output data.jsonl --format jsonl
```

### GUI Interface Overview

The application features a modern tabbed interface with the following sections:

#### 📄 Convert Tab

- Load existing datasets from various formats
- Apply template-based conversions (instruction, chat, Q&A)
- Interactive field mapping with smart suggestions
- Preview converted data before saving
- Export to multiple formats

#### ✨ Create Tab

- Generate datasets from scratch using templates
- Configurable number of records
- AI-powered field generation using Faker
- Custom field types and patterns
- Real-time preview of generated data

#### ✏️ Edit Tab

- Direct table editing of dataset records
- Add/remove records dynamically
- Search and filter capabilities
- Bulk operations and data manipulation
- Save changes in any supported format

#### ✅ Validate Tab

- Comprehensive data validation
- Missing value detection
- Type consistency checking
- Duplicate record identification
- Custom validation rules
- Detailed validation reports

#### 📊 Visualize Tab

- Dataset statistics and insights
- Interactive charts and graphs
- Field distribution analysis
- Missing value visualization
- Export statistics and charts

## 🔧 Configuration

### Directory Structure

The application automatically creates the following directories:

```
dataset_converter/
├── dataset_input/     # Input files directory
├── dataset_output/    # Output files directory
├── dataset_logs/      # Application logs
└── ...
```

### Supported Input Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON | `.json` | Single JSON object or array |
| JSONL | `.jsonl` | JSON Lines (one JSON per line) |
| CSV | `.csv` | Comma-separated values |
| TSV | `.tsv` | Tab-separated values |
| Excel | `.xlsx`, `.xls` | Microsoft Excel files |
| Parquet | `.parquet` | Apache Parquet columnar format |
| HDF5 | `.h5`, `.hdf5` | Hierarchical Data Format |
| SQLite | `.db`, `.sqlite` | SQLite database files |
| XML | `.xml` | Extensible Markup Language |
| YAML | `.yaml`, `.yml` | YAML Ain't Markup Language |
| Avro | `.avro` | Apache Avro serialization |
| TXT | `.txt` | Plain text files |

### Dataset Templates

#### Instruction Template

```json
{
  "prompt": "User instruction or question",
  "completion": "Expected response or answer"
}
```

#### Chat Template

```json
{
  "messages": [
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

#### Q&A Template

```json
{
  "question": "Question text",
  "answer": "Answer text",
  "context": "Optional context information"
}
```

## 🔍 Features Deep Dive

### Data Validation

- **Missing Values**: Detects empty, null, or whitespace-only fields
- **Type Consistency**: Ensures field types are consistent across records
- **Duplicate Detection**: Identifies exact and near-duplicate records
- **Custom Rules**: Regular expressions, range checks, required fields
- **Format Validation**: Validates data against template structures

### Data Augmentation

- **Text Rephrasing**: Synonym replacement and sentence restructuring
- **Data Expansion**: Generate variations of existing records
- **Noise Addition**: Add controlled randomness for robustness
- **Balancing**: Oversample underrepresented classes

### Data Anonymization

- **PII Detection**: Automatic detection of personally identifiable information
- **Faker Integration**: Replace sensitive data with realistic fake data
- **Custom Patterns**: Define custom anonymization rules
- **Preservation**: Maintain data utility while ensuring privacy

### Visualization & Analytics

- **Dataset Overview**: Record counts, field distributions, basic statistics
- **Missing Data Analysis**: Heatmaps and patterns of missing values
- **Field Analysis**: Value distributions, unique counts, data types
- **Correlation Analysis**: Relationships between numeric fields
- **Export Options**: Save charts and statistics in multiple formats

## 🧪 Testing

### Running Tests

```bash
# Run PowerShell validation scripts
.\file_finalizer_check.ps1 -Path . -Recurse
.\universaL_syntax_checker.ps1 -Path . -Recurse  
.\universal_error_checker.ps1 -Path . -Recurse

# Python syntax validation
python -m py_compile dataset_generator_original.py
```

### Test Scripts

- **file_finalizer_check.ps1**: Final pre-deployment validation
- **universaL_syntax_checker.ps1**: Multi-language syntax validation
- **universal_error_checker.ps1**: Error pattern detection

## 🐛 Known Issues

### Critical Issues

- PyQt5 attribute compatibility issues with newer versions
- Type annotation inconsistencies in function parameters
- Optional member access without null checking

### Medium Priority

- Large dataset performance optimization needed
- GUI thread blocking during long operations
- Memory usage optimization for very large files

See [TODO.md](TODO.md) for complete issue tracking and roadmap.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 dataset_generator_original.py

# Run type checking
mypy dataset_generator_original.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyQt5** for the excellent GUI framework
- **pandas** for powerful data manipulation capabilities
- **Rich** for beautiful console output
- **Faker** for realistic data generation
- **matplotlib/seaborn** for visualization capabilities

## 📞 Support

- 📧 **Issues**: Use GitHub Issues for bug reports and feature requests
- 📚 **Documentation**: Check the [TODO.md](TODO.md) for current development status
- 💬 **Discussions**: Use GitHub Discussions for questions and community support

## 🔮 Roadmap

### Version 2.1 (Next Release)

- [ ] Fix critical PyQt5 compatibility issues
- [ ] Implement comprehensive error handling
- [ ] Add unit test framework
- [ ] Performance optimization for large datasets

### Version 2.2 (Future)

- [ ] Plugin architecture for custom formats
- [ ] Advanced ML-based data validation
- [ ] Real-time collaboration features
- [ ] Cloud storage integration

### Version 3.0 (Long-term)

- [ ] Web-based interface option
- [ ] Distributed processing support
- [ ] Advanced AI-powered data generation
- [ ] Enterprise features and deployment options

---

**Made with ❤️ for the LLM fine-tuning community**
