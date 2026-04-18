# Autonomous Car Racing Agent 🏎️🤖

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-standard%20library-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An autonomous agent ("bot") designed to compete in a multi-agent, physics-based car racing environment. The bot navigates a 2D grid under a "fog of war," autonomously exploring the map and navigating to goal cells using foundational AI and robotics algorithms.

## ✨ Features

* **Physics-Based Kinematics:** Handles acceleration vectors, velocity, momentum, and inertia to ensure smooth, crash-free navigation.
* **A* Strategic Pathfinding:** Utilizes A* search with a Manhattan distance heuristic to find the most efficient route across known terrain.
* **Frontier-Based Exploration:** Systematically discovers unknown areas of the map using a Breadth-First Search (BFS) variant, commonly used in SLAM.
* **Model Predictive Control (MPC):** Employs a limited-horizon lookahead controller to evaluate tactical movements, prune unsafe moves, and optimize trajectory based on speed, safety, and momentum.
* **Zero External Dependencies:** Built entirely "by hand" using Python's standard library (`heapq`, `collections.deque`).

## 🧠 AI Architecture

The agent's decision-making pipeline is split into strategic and tactical layers:

1. **State Space Representation:** The map is stored as a highly memory-efficient sparse dictionary (`Dict[Tuple[int, int], int]`) to perfectly model epistemic uncertainty (the "fog of war").
2. **Exploration vs. Exploitation:** * If the goal is unknown, the bot seeks the nearest "frontier" boundary to maximize information gain.
   * If the goal is known, the bot plots an optimal grid-path using A*.
3. **Tactical Movement:** The lookahead controller evaluates the 9 possible acceleration inputs $a_{t}=(a_{r},a_{c})$ over a short horizon. It uses a composite cost function weighing:
   * Distance to target
   * Wall proximity (safety margin)
   * Speed constraints
   * Acceleration smoothness (anti-jerk)
   * Heading alignment (momentum conservation)

## 🚀 Getting Started

### Prerequisites
* Python 3.6 or higher.
* No external libraries (e.g., `numpy` or `scipy`) are required.
