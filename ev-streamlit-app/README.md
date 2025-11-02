# EV Streamlit App

## Overview
The EV Streamlit App is a web application designed for cleaning and visualizing electric vehicle (EV) datasets. Users can upload their datasets, perform data cleaning operations, and visualize the results through an intuitive interface.

## Project Structure
```
ev-streamlit-app
├── src
│   ├── app.py                     # Main entry point of the Streamlit application
│   ├── components
│   │   └── sidebar.py             # Sidebar navigation components
│   ├── pages
│   │   ├── data_cleaning.py       # Data cleaning interface
│   │   └── visualization.py        # Visualization interface
│   ├── data_preprocessing.py       # Data preprocessing functions
│   ├── visualization_helpers.py     # Helper functions for visualizations
│   └── styles
│       └── styles.css             # Custom CSS styles
├── .streamlit
│   └── config.toml                # Streamlit configuration settings
├── requirements.txt               # Project dependencies
├── .gitignore                     # Files to ignore in Git
└── README.md                      # Project documentation
```

## Features
- **Data Cleaning**: Upload CSV files, preview data, and apply automatic cleaning processes.
- **Data Visualization**: Generate various visualizations based on the cleaned datasets.
- **Custom Styling**: Enhanced visual presentation through custom CSS.

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ev-streamlit-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage
- Navigate to the data cleaning page to upload and clean your datasets.
- Switch to the visualization page to create and view visualizations based on the cleaned data.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.