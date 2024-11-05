# Frame-Based Monitoring

<!--- BADGES: START --->

![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)
![PyPI](https://img.shields.io/pypi/v/lotus-ai)
[![GitHub license](https://img.shields.io/badge/License-MIT-blu.svg)][#license-gh-package]

[#license-gh-package]: https://lbesson.mit-license.org/

<!--- BADGES: END --->

Overview:
The Frame-Based Monitoring project is a powerful toolkit for parsing, condensing, and analyzing frame data with a focus on enhancing efficiency, interpretability, and actionable insights. Built around a modular workflow, this project enables applications in autonomous systems, symbolic processing, and natural language processing (NLP) through the use of static frame condensation, symbolic parsing, and trajectory analysis.

Key features of this project include:

Static Frame Condensation: The project introduces a StaticCondenser module that aggregates multiple frames into a single, condensed frame. By averaging or combining frame attributes (e.g., position, velocity), this condensation process reduces data redundancy while preserving essential scene information. Condensed frames help streamline computations and enable efficient data handling for downstream applications.

Symbolic Interpretation: Using a SymbolicParser, this toolkit converts raw numerical data from frames into symbolic representations, making the data more interpretable for real-time analysis and decision-making. This symbolic layer bridges the gap between complex data inputs and human-readable formats, supporting explainable AI applications.

Trajectory Analysis: The toolkit provides tools to track object movement across frames, computing trajectories that predict future object states. This is especially valuable in autonomous systems, where understanding object paths and motion trends is crucial for predictive decision-making.

The Frame-Based Monitoring project supports applications in fields like autonomous driving, robotics, and AI-driven analytics by enabling real-time data optimization, symbolic representation, and predictive analysis. By consolidating complex data into condensed and interpretable formats, this repository provides a foundation for developing scalable, efficient, and explainable AI solutions.

## Key Concept: Static Frame Condensation
This project focuses on condensing and analyzing frames, initially handling static frames for now. This system aims to streamline data from various sources (e.g., NuScenes dataset) by extracting symbolic representations, which can be applied in autonomous systems and object state prediction.

## Goals
1. **Condense Frames:** Transform dense, complex data into simplified symbolic representations for easier processing.
2. **Dynamic Frame Condensation (Future):** Build upon static frame condensation to enable real-time data updates and adaptive frame structures.
3. **Symbolic Parsing:** Use a symbolic parser to convert data into human-readable text, enhancing interpretability.

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
