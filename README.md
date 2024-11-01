# Frame-Based Monitoring

<!--- BADGES: START --->

![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)
![PyPI](https://img.shields.io/pypi/v/lotus-ai)
[![GitHub license](https://img.shields.io/badge/License-MIT-blu.svg)][#license-gh-package]

[#license-gh-package]: https://lbesson.mit-license.org/

<!--- BADGES: END --->

## Key Concept: Static Frame Condensation
This project presents a frame condensation system that processes large, complex datasets, primarily focusing on data relevant to autonomous systems. The system condenses data from individual frames and explores symbolic representation to simplify and enhance the usability of this information for tasks like object state prediction and environmental analysis.

Initially, this project focuses on static frame condensation, where each frame is individually analyzed and simplified into a symbolic form that captures essential details. These static representations act as the foundation for subsequent steps, allowing for a high-level understanding of the scene without requiring every detail.

By leveraging data from sources such as the NuScenes dataset, the system condenses information into symbolic forms, facilitating easier data interpretation and efficient computational processing. Through the symbolic representation of frame data, this project aims to support object tracking, motion prediction, and obstacle avoidance in real-world applications where quick, adaptive responses are crucial.

## Goals
1. **Condense Frames:** Transform dense, complex data into simplified symbolic representations for easier processing.
2. **Dynamic Frame Condensation:** Build upon static frame condensation to enable real-time data updates and adaptive frame structures.
3. **Symbolic Parsing:** Use a symbolic parser to convert data into human-readable text, enhancing interpretability.

## Folder Structure

- **`src/condensing`**: Contains the primary code for condensing static frames.
- **`src/symbolic_parser`**: Houses files for parsing and converting frames into symbolic representations.
- **`dynamic_frame_condensation`**: Future work that will expand upon static frame condensation to enable dynamic, real-time updates.


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

- Nuscenes dataset download


- Initialize the condensor with your data
condensor = StaticFrameCondensor(data)

- Run the condensation process
condensed_data = condensor.condense()

- Save or analyze the condensed data
save(condensed_data, 'output_file_path')

Input:
Frame structure

Output: 
Frames --

#Examples

One scene of the dataset --> How the frame gets condensed

#Condensation
Contributions are welcome! Please fork this repository, create a branch, and submit a pull request with your changes.
