LLM-Powered Evidence Triangulation Framework
This project is a powerful and flexible Python framework designed to streamline the process of systematic literature reviews. It automates the extraction of key concepts from scientific abstracts by leveraging multiple Large Language Models (LLMs). By running the same task across different models, the framework enables evidence triangulation, a method used to cross-validate findings and improve the reliability of your data.

Why Use This Framework?
Data extraction from academic papers is typically a time-consuming and manual process. This framework addresses that challenge by offering several key benefits:

Automation: Automatically extract structured data, such as exposures and outcomes, from a large corpus of scientific abstracts.

Triangulation for Reliability: Compare the results from various LLMs (like DeepSeek, GLM, or Qwen) to identify areas of agreement and disagreement. This "triangulation" process helps you build confidence in the extracted data and provides a robust way to handle potential model biases.

Performance Analysis: The final consolidated output allows you to easily analyze and compare the performance of each model, helping you determine which ones are most effective for your specific task.

Project Structure
The repository is organized into a clear and logical structure to make it easy to navigate:

main.py: This is the central control script. It orchestrates the entire workflow from start to finish.

extraction_matching.py: Contains the core logic for API calls to the LLMs and the data processing that formats the extracted information.

triangulation.py: A script designed for analyzing and visualizing the results. You can use it to generate performance metrics, such as precision, recall, and F1 scores, and create visualizations.

data/: This directory is for your input and output files.

all_got_df_final_step_2_salt_cvd_021025.xlsx: The sample input file containing the scientific abstracts.

combined_extraction_results.xlsx: The final output file with the consolidated results from all LLMs.

requirements.txt: Lists all the Python packages required to run the project.

.env.example: A template for your environment file, which securely stores your API keys.

Getting Started
Follow these steps to set up and run the project on your local machine.

1. Environment Setup
It is highly recommended to use a virtual environment to manage the project's dependencies and prevent conflicts with other Python projects.

Create the virtual environment:
Open your terminal or command prompt in the project's root directory and run:

python -m venv .venv

Activate the virtual environment:

On macOS and Linux:

source .venv/bin/activate

On Windows:

.venv\Scripts\activate

2. Installation
With your virtual environment active, install the necessary Python libraries using pip:

pip install -r requirements.txt

3. Configuration
To use the LLM APIs, you need to provide your API key.

Create a new file named .env in the root of your project directory.

Copy the contents of the .env.example file into your new .env file.

Replace YOUR_API_KEY with your actual key. This single key is used across all configured LLMs.

4. Running the Project
Once you have completed the setup, you can run the main script from your terminal:

python main.py

The script will display its progress and save the final consolidated results to the combined_extraction_results.xlsx file in the data/ directory.

Example of Triangulation
The core of this framework is to leverage the unique strengths of multiple LLMs. For instance, if you are extracting exposure and outcome pairs from abstracts, a single model might miss a key term. By using three different models, you can triangulate the results.

PMID

Extracted from Model A

Extracted from Model B

Extracted from Model C

1234

sodium intake

high salt diet

sodium intake

5678

hypertension

blood pressure

hypertension

In this example, Models A and C agree on the exposure, which gives you higher confidence in the result. Model B's different phrasing (high salt diet) still provides valuable, related information. This collective data provides a richer, more reliable dataset than any single model could on its own.