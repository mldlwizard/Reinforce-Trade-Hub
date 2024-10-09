# Reinforce Trade Hub

## Human-Centric AI Framework for Optimizing Trading Portfolios using Deep Reinforcement Learning

Author: Yalala Mohit
Supervisor: Dr. Stacy Marsella

[![GitHub Repository](https://img.shields.io/badge/github-mldlwizard/Reinforce--Trade--Hub-blue.svg)](https://github.com/mldlwizard/Reinforce-Trade-Hub)

## Project Overview

Reinforce Trade Hub is an innovative project that leverages Deep Reinforcement Learning (DRL) to optimize trading portfolios. The framework incorporates transformers and actor-critic methodology to dynamically select stocks and optimize trading actions within a simulated market environment. This project aims to provide robust strategy formulation for trading decisions.

### Key Features

1. Deep Reinforcement Learning agent for portfolio optimization
2. Integration of transformers and actor-critic methods
3. Simulated market environment for strategy testing
4. Streamlit app deployed on Google Cloud Platform (GCP)
5. Integration with Candle Charts for visual analysis
6. Large Language Model (GPT-4) integration for narrative explanations of trading decisions
7. Empirical study conducted with 50 participants

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Models](#models)
5. [Web Interface](#web-interface)
6. [Data Visualization](#data-visualization)
7. [Empirical Study](#empirical-study)
8. [Contributing](#contributing)
9. [License](#license)

## Installation

To set up the Reinforce Trade Hub project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/mldlwizard/Reinforce-Trade-Hub.git
   cd Reinforce-Trade-Hub
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized into the following main directories:

- `src/`: Contains the core implementation of the DRL models.
- `Supplementary/`: Includes additional scripts for data processing, visualization, and web interface.

Key files in the project:

- `src/ddpg.py`: Implementation of the Deep Deterministic Policy Gradient algorithm.
- `src/dqn.py`: Implementation of the Deep Q-Network algorithm.
- `Supplementary/frontend.py`: Streamlit-based web interface for the trading platform.
- `Supplementary/candlechart.py`: Script for generating candlestick charts.
- `Supplementary/LLMexplanability.py`: Integration with GPT-4 for generating narrative explanations.

## Usage

To run the main trading simulation:

```
python src/main.py
```

To launch the Streamlit web interface:

```
streamlit run Supplementary/frontend.py
```

## Models

The project implements two main reinforcement learning models:

1. **Deep Deterministic Policy Gradient (DDPG)**
   - File: `src/ddpg.py`
   - Purpose: Continuous action space for fine-grained trading decisions.

2. **Deep Q-Network (DQN)**
   - File: `src/dqn.py`
   - Purpose: Discrete action space for buy, sell, and hold decisions.

Both models are designed to interact with the simulated market environment and learn optimal trading strategies over time.

## Web Interface

The project features a web-based interface built with Streamlit:

- File: `Supplementary/frontend.py`
- Features:
  - User authentication
  - Portfolio creation and management
  - Stock recommendations using the trained DRL models
  - Interactive data visualization

To access the interface, run the Streamlit command mentioned in the [Usage](#usage) section.

## Data Visualization

Reinforce Trade Hub incorporates advanced data visualization techniques:

1. **Candlestick Charts**
   - File: `Supplementary/candlechart.py`
   - Purpose: Visualize stock price movements and trends.

2. **Interactive Plots**
   - File: `Supplementary/candlechartstreamlit.py`
   - Purpose: Allow users to explore stock data with customizable date ranges and stock selections.

## Empirical Study

An empirical study was conducted with 50 participants to evaluate the effectiveness of the Reinforce Trade Hub framework. The study assessed various aspects such as:

- User experience with the web interface
- Effectiveness of the AI-driven recommendations
- Comprehension of the narrative explanations provided by the LLM
- Overall impact on trading decisions and portfolio performance

Detailed results and analysis of the empirical study can be found in the project documentation.

## Contributing

Contributions to Reinforce Trade Hub are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request


For more information about the project, please contact Yalala Mohit or refer to the [GitHub repository](https://github.com/mldlwizard/Reinforce-Trade-Hub).
