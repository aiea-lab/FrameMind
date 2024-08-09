NuScenes Frame-Based Monitoring Experiment

This repository contains the implementation and results of an experiment conducted on the NuScenes dataset, aligning with the objectives outlined in the AFOSR YIP proposal. The experiment leverages a frame-based monitoring system to detect, diagnose, and explain errors in multimodal autonomous systems, particularly focusing on scenarios depicted in the NuScenes dataset.

Objectives
The primary objective of this experiment is to develop and validate a frame-based monitoring system for multimodal autonomous systems. This work is inspired by the hypothesis that a symbolic, frame-based representation can enhance the precision and explainability of error detection in complex, real-world scenarios.

Experiment Structure
1. Frame Structure Implementation
Representation: We translated multimodal log data from the NuScenes dataset into a symbolic frame-based representation.
Integration: The representation was integrated with a reasoning engine that compares and diagnoses errors in autonomous driving scenarios.
2. NuScenes Dataset
Initialization: The NuScenes dataset was initialized, and specific samples were selected for analysis.
Visualization: Frames from the dataset were visualized, focusing on sensor data such as camera images and LiDAR point clouds.
3. Experimentation
Setup: A series of experiments were conducted to validate the frame-based monitoring system's effectiveness in real-world autonomous driving scenarios.
Results: The outcomes were analyzed to determine the system's precision, recall, and ability to provide symbolic explanations for detected errors.

Requirements

- Python 3.x
- Conda environment with the necessary dependencies

How to Run the Experiment
Prerequisites
Python 3.x
Conda environment with the following packages:
nuscenes-devkit
matplotlib
opencv-python
Installation
Clone the Repository
Create and Activate the Conda Environment

Running the Experiment
Initialize the NuScenes Dataset:

Update the main.py file with the correct path to your NuScenes dataset.

Run the Frame Visualization Script:

Execute the script to visualize and diagnose frames:

View and Analyze Results:

The results will include visualized frames and diagnostic information, which can be used to analyze the system's performance.
Experiment Results
The experiment successfully demonstrated the feasibility of using a frame-based monitoring system for multimodal autonomous systems. Key results include:

Improved Precision: The frame-based approach showed higher precision in error detection compared to traditional methods.
Enhanced Explainability: The system provided symbolic explanations that were both intuitive and aligned with human reasoning, increasing user trust.

Acknowledgments
This work is part of the research supported by the AFOSR Young Investigator Program (YIP). The NuScenes dataset used in this experiment is provided by Motional.

License
This project is licensed under the MIT License. See the LICENSE file for details.