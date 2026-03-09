# Autonomous Perception & Planning Pipeline

End-to-end autonomous driving system in CARLA — camera input to vehicle control using semantic segmentation and path planning.

> **Work in Progress** — actively being built.

## Overview

The system processes camera images from a simulated vehicle through a 5-stage pipeline:

```
Camera → Segmentation (U-Net) → Occupancy Grid → Path Planning (A*) → Vehicle Control (PID)
```

Each stage runs as a separate ROS2 node, containerized with Docker.

## Tech Stack

- **Simulation:** CARLA 0.9.16
- **Perception:** PyTorch, U-Net with ResNet-50 encoder
- **Planning:** Custom A* on occupancy grid, cubic spline smoothing
- **Control:** PID controller
- **Infrastructure:** ROS2 Humble, Docker, GitHub Actions

## Roadmap

- [ ] Environment setup (CARLA + ROS2 + Docker)
- [ ] Data collection (8K+ labeled frames from CARLA)
- [ ] Semantic segmentation model (target mIoU > 70%)
- [ ] Path planning on occupancy grid
- [ ] Full pipeline integration
- [ ] Documentation & demo

## Requirements

- Windows 11, NVIDIA GPU (RTX 2060+)
- WSL2 with Ubuntu 22.04
- Docker with NVIDIA Container Toolkit
- CARLA 0.9.16

## Author

Olti Rexhepaj — Applied CS @ [HTWG Konstanz](https://www.htwg-konstanz.de/)

## License

MIT