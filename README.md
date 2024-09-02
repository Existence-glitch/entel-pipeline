# EntelRecon Retrieval-Augmented Generation Pipeline

## Project Overview
[Provide a brief description of your project, its goals, and its significance in the context of cybersecurity and machine learning]

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Pipeline Stages](#pipeline-stages)
- [Models](#models)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.10
- Conda

### Setup
1. Clone the repository:
   ```
   git clone [Your Repository URL]
   cd entel-pipeline
   ```

2. Create and activate the conda environment:
   ```
   conda create -n entel-pipeline python=3.10
   conda activate entel-pipeline
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
entel-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── database_config.yaml
│   └── model_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── docs/
│   └── project_documentation.md
├── models/
│   ├── trained/
│   └── pretrained/
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── embedding.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rag_model.py
│   │   └── risk_classifier.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
└── tests/
    ├── __init__.py
    ├── test_ingestion.py
    ├── test_preprocessing.py
    └── test_models.py
```


## Usage
[Provide instructions on how to use your pipeline, including any command-line interfaces or scripts]

## Data
[Describe your data sources, format, and any preprocessing steps]

### Data Format
Our pipeline processes cybersecurity detection data in JSON format. Each detection entry includes the following fields:
- `did`: Unique identifier for the detection
- `ip`: IP address associated with the detection
- `fqdn`: Fully Qualified Domain Name
- `puerto`: Port number
- `fingerprint`: Fingerprint details
- `data`: Additional detection data
- `cpe`: Common Platform Enumeration (if available)
- `cve`: Common Vulnerabilities and Exposures (if available)
- `vectores`: Associated attack vectors

## Pipeline Stages
[Describe each stage of your pipeline, including data ingestion, preprocessing, embedding generation, and model training]

### 1. Data Ingestion

The data ingestion stage is responsible for loading the raw cybersecurity detection data into our pipeline. This stage is implemented in the `src/data/ingestion.py` file.

Key features of the data ingestion stage:
- Reads JSON files containing cybersecurity detection data.
- Supports ingestion of single files or entire directories.
- Defines a schema for the data, ensuring consistency and proper data typing.
- Preprocesses the data to handle potential inconsistencies in the raw data.
- Creates a PySpark DataFrame for efficient data processing.

The `DataIngestion` class provides the following main methods:
- `ingest_file(file_path)`: Ingests a single JSON file.
- `ingest_directory(directory_path)`: Ingests all JSON files in a directory.
- `create_dataframe(data)`: Creates a PySpark DataFrame from the ingested data.
- `ingest_and_create_dataframe(path)`: Combines ingestion and DataFrame creation.

### 2. Data Preprocessing

The data preprocessing stage prepares the ingested data for embedding generation and further analysis. This stage is implemented in the `src/data/preprocessing.py` file.

Key features of the data preprocessing stage:
- Cleans and normalizes text fields.
- Handles missing values in the dataset.
- Converts IP addresses to a format suitable for text processing.
- Combines relevant fields into a single text representation for each record.

The `DataPreprocessor` class provides the following main methods:
- `preprocess(df)`: Applies all preprocessing steps to the input DataFrame.
- `clean_text(text)`: Cleans and normalizes individual text fields.
- `combine_fields(...)`: Combines relevant fields into a single text representation.

Preprocessing steps include:
1. Text cleaning: Convert to lowercase, remove special characters (preserving important ones for version numbers and CPEs), and remove extra whitespace.
2. Handling missing values: Fill null values for 'fqdn' and 'cpe' fields.
3. IP address conversion: Replace dots with underscores in IP addresses.
4. Port number conversion: Convert 'puerto' (port) to string type for consistency.
5. Text combination: Create a 'combined_text' field that includes all relevant information from each record.

The preprocessed data is ready for the next stage of the pipeline, which will involve embedding generation.

### 3. Embedding Generation
### 4. Indexing and Retrieval
### 5. RAG Model Development
### 6. Evaluation and Iteration



## Models
[Describe the models used in your pipeline, including any pre-trained models and custom models you've developed]

## Testing
[Explain how to run tests and what they cover]

## Contributing
[Guidelines for contributing to the project]

## License
[Specify the license under which your project is released]

---

[Optional: Add badges for build status, code coverage, etc.]

```

[Remember to replace placeholder text with actual project details as you progress]
```
