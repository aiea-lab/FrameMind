# Frame-Based Monitoring

<!--- BADGES: START --->

![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)
![PyPI](https://img.shields.io/pypi/v/lotus-ai)
[![GitHub license](https://img.shields.io/badge/License-MIT-blu.svg)][#license-gh-package]

[#license-gh-package]: https://lbesson.mit-license.org/

<!--- BADGES: END --->

Overview
The Frame-Based Monitoring project provides tools for condensing and analyzing frame data, focusing on static frame condensation. Initially designed to handle static frames, this project helps optimize and simplify data for applications in autonomous systems, symbolic parsing, and natural language processing (NLP). It also enables real-time object state prediction and decision-making by using condensed frame data.

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
