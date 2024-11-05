# Frame-Based Monitoring

<!--- BADGES: START --->

![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)
![PyPI](https://img.shields.io/pypi/v/lotus-ai)
[![GitHub license](https://img.shields.io/badge/License-MIT-blu.svg)][#license-gh-package]

[#license-gh-package]: https://lbesson.mit-license.org/

<!--- BADGES: END --->

## Overview:
The **Frame-Based Monitoring** project is a sophisticated toolkit designed for advanced parsing, condensation, and analysis of high-dimensional frame data, with a particular emphasis on static frame condensation. This research-driven framework addresses core challenges in data optimization, interpretability, and real-time decision-making, providing a modular approach to frame-based data processing. Its applications span autonomous systems, symbolic processing, and natural language processing (NLP), enabling foundational research into efficient data handling and symbolic AI.

## Research Objectives

### Data Optimization via Frame Condensation
The project introduces a **Static Frame Condensation** module to aggregate multiple frames into a singular, representative frame, reducing data redundancy while retaining essential scene dynamics. The condensation process leverages statistical aggregation (e.g., averaging positions and velocities) to synthesize a condensed representation that supports computational efficiency. This work aims to demonstrate that condensed frames can maintain fidelity to the original scene while significantly minimizing data volume, a key advancement for data-intensive applications.

### Symbolic Parsing for Enhanced Interpretability: 
A symbolic interpretation layer, operationalized through a Symbolic Parser, converts raw numerical frame data into structured symbolic representations. This transformation allows data to be rendered in a human-readable format, facilitating interpretability and bridging the gap between machine-generated data and human-centered analysis. Symbolic parsing is particularly impactful in AI explainability, supporting applications where transparency and interpretability of data-driven insights are paramount.

### Predictive Analysis through Trajectory Computation: 
The project integrates a Trajectory Analysis component to extract object trajectories across frames, providing insights into object state evolution over time. By tracking the motion paths of objects within a scene, this module supports predictive analytics in applications such as autonomous driving, where accurate motion forecasting is critical. This work investigates how trajectory modeling can enhance both the predictive power and reliability of real-time decision systems.

## Contribution to the Field

The **Frame-Based Monitoring** project offers a structured, modular approach to high-dimensional frame data processing, with applications across autonomous systems, robotics, and AI-driven analytics. Through innovations in static frame condensation, symbolic parsing, and trajectory analysis, this project establishes a framework that advances both computational efficiency and data interpretability.

By addressing scalability, symbolic AI, and real-time predictive modeling, this repository provides a foundation for future developments in adaptive frame condensation, explainable AI, and autonomous decision-making systems. The **Frame-Based Monitoring** research project represents a versatile framework for further research and practical advancements in efficient data representation, symbolic processing, and predictive analytics in AI and autonomous systems.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/frame-based-monitoring.git
   cd frame-based-monitoring
   ```

2. **Install Dependencies**
   Make sure you have Python 3.7+ installed, then install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**
   - Download the NuScenes dataset or another compatible dataset and place it in the designated `data` folder.
   - Follow any specific instructions for formatting the data for use in the project.

## Folder Structure

- **`src/condensing`**: Contains the primary code for condensing static frames.
- **`src/symbolic_parser`**: Houses files for parsing and converting frames into symbolic representations.
- **`dynamic_frame_condensation`**: Future work that will expand upon static frame condensation to enable dynamic, real-time updates.

## Applications

- **Autonomous Systems**: Enhancing decision-making by reducing the volume of static data frames processed, thereby optimizing computational resources.
- **Natural Language Processing (NLP)**: Improving text condensation and summarization tasks by filtering repetitive frame data in symbolic language models.

The repository contains tools and methods for:
- **Identifying Redundant Frames**: Employing statistical analysis and clustering techniques to detect duplicate or similar frames from static datasets.
- **Optimized Data Structures**: Utilizing efficient data structures that facilitate condensation of static frames, enabling faster processing speeds.
- **Real-World Applications**: With applications in AI, robotics, and NLP, this project serves as a foundation for further research into frame-based communication, decision-making models, and symbolic processing.

The main goal of this repository is to provide a robust framework for researchers and developers working on frame condensation techniques, with a focus on both efficiency and effectiveness in data handling.

#Installation
- Clone this repository
git clone https://github.com/aiea-lab/frames.git

- Navigate to the project directory
cd scripts/nuscenes_main.py

- Install dependencies
pip install -r requirements.txt

#Usage
- Import the static frame condensation module
from condensation import StaticFrameCondensor

- Initialize the condensor with your data
condensor = StaticFrameCondensor(data)

- Run the condensation process
condensed_data = condensor.condense()

- Save or analyze the condensed data
save(condensed_data, 'output_file_path')

#Examples
We are using the Boston Scene for the parsing process
Parsed Frames:
Frame(name="Frame1", slots={'position': [1.0, 2.0, 3.0], 'velocity': [0.5, 0.0, 1.0], ...})
Frame(name="Frame2", slots={'position': [1.5, 2.5, 3.5], 'velocity': [0.6, 0.1, 0.9], ...})

Symbolic Frames:
SymbolicFrame(name="Frame1", data={'position_symbol': "Pos1", 'velocity_symbol': "Vel1", ...})
SymbolicFrame(name="Frame2", data={'position_symbol': "Pos2", 'velocity_symbol': "Vel2", ...})

Condensed Frame Data:
{
    'position': [1.25, 2.25, 3.25],
    'velocity': [0.55, 0.05, 0.95],
    ...
}

Object Trajectories:
Object ID: obj_1, Trajectory: [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], ...]
Object ID: obj_2, Trajectory: [[2.0, 1.0, 0.5], [2.5, 1.5, 1.0], ...]


#Condensation
Contributions are welcome! Please fork this repository, create a branch, and submit a pull request with your changes.
