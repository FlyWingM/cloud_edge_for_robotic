# Adaptive Resource Management for Cloud-Edge Robotics
Overview
This repository contains a cloud-native implementation of adaptive resource management for latency-sensitive cloud robotics applications. Our approach integrates dynamic workload allocation, intelligent model selection, and automated fault detection and recovery to mitigate long-tail latency anomalies (P99, P95) in hybrid cloud-edge environments.

The system dynamically distributes computational workloads between high-accuracy, cloud-based models (e.g., Faster R-CNN, Mask R-CNN) and low-latency, edge-deployed models (e.g., YOLO series). It ensures robust performance under resource fluctuations, network variability, and unexpected failures.

Key Features

✔ Cloud-Native Deployment: Uses Kubernetes and containerized microservices for scalable orchestration of AI workloads.

✔ Hybrid Cloud-Edge Adaptation: Dynamically offloads inference tasks between cloud and edge nodes based on accuracy-latency trade-offs.

✔ Adaptive Model Switching: Selects optimal AI models based on real-time system constraints and workload conditions.

✔ Long-Tail Latency Mitigation: Reduces worst-case latency spikes (P99, P95) with predictive analytics and intelligent scheduling.

✔ Fault-Tolerant Execution: Implements automated failure detection, dynamic task reallocation, and redundant execution strategies.

✔ Energy & Cost Efficiency: Optimizes resource allocation to reduce carbon footprint and operational costs in cloud deployments.

System Architecture
Our cloud-native system is structured as follows:

Kubernetes-Based Orchestration:

Deploys workloads as containerized services.
Uses horizontal pod autoscaling for adaptive inference scaling.
Implements KubeEdge for seamless cloud-edge task management.
Edge-Cloud AI Workload Management:

Runs lightweight inference on edge nodes for low-latency tasks.
Sends complex, accuracy-critical workloads to cloud-based AI models.
Implements model-aware adaptive switching to optimize response time.
Intelligent Resource Allocation:

Predicts workload demand and optimally assigns computational resources.
Uses reinforcement learning and predictive modeling to adapt allocation policies.
Dynamically reconfigures inference pipelines for real-time performance.
Automated Failure Handling:

Distributed anomaly detection for identifying latency spikes, network delays, and system failures.
Redundant execution strategies for mission-critical AI workloads.
Dynamic fault recovery mechanisms to ensure continuous service.

### **Enable Dynamic Offloading**
To enable dynamic offloading in the Kubernetes cluster, run:

```bash
kubectl apply -f deployment/offload-controller.yaml

