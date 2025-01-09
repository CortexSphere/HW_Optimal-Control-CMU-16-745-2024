# HW Optimal Control (CMU 16-745, 2024)

[![GitHub last commit](https://img.shields.io/github/last-commit/CortexSphere/HW_Optimal-Control-CMU-16-745-2024)](https://github.com/CortexSphere/HW_Optimal-Control-CMU-16-745-2024/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/CortexSphere/HW_Optimal-Control-CMU-16-745-2024)](https://github.com/CortexSphere/HW_Optimal-Control-CMU-16-745-2024)

This repository contains homework and assignments for the **Optimal Control** course (CMU 16-745, Spring 2024). It focuses on numerical methods, system modeling, and optimal control strategies.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Problem](#Problem)
---

## Introduction

This repository includes various homework problems and projects related to **optimal control**. It covers topics such as:
- Quadratic programming (QP)
- Liner Quadratic Regulator (LQR)
- Model predictive control (MPC)
- Control of dynamic systems

---

## Problem

## 1. Falling Brick
This module implements a simulation of a falling brick sliding on ice. The problem is formulated as a Quadratic Programming (QP) problem with equality and inequality constraints. 

- **Core Topics**: Augmented Lagrangian Method, Quasi-Newton Optimization
- **Key Highlights**:
  - Solves QP with constraints using an augmented Lagrangian framework.
  - Implements numerical gradient and mask matrices for inequality constraints.
  - Includes visualization of the brick's trajectory over time.
- **Path**: `falling_brick/`
  - `falling_brick_simulation.py`: Main script to simulate the falling brick and solve the QP problem.

---

## 2. 2D Drone
This module focuses on controlling a planar quadrotor to move towards a fixed point while respecting constraints. It uses Convex Model Predictive Control (MPC) to solve a constrained Linear Quadratic Regulator (LQR) problem.

- **Core Topics**: Convex MPC, Constrained LQR
- **Key Highlights**:
  - Simulates 2D quadrotor dynamics.
  - Implements both LQR and MPC controllers.
  - Solves the constrained LQR problem to achieve smooth, stable control towards the target.
  - Includes animated visualizations of the quadrotor's trajectory.
- **Path**: `2D_drone/`
  - `quadrotor_simulation.py`: Main script to simulate quadrotor control using MPC and LQR.
