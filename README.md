The file `kuramoto_3subplots_N20` is for a system with N=20 oscillators, while the file `kuramoto_3subplots_N200` is for a system with N=200 oscillators.
N=200.gif contains the full simulation used to generate the snapshots shown in the manuscript for N=200 oscillators.
N=20.gif shows the simulation video from which the manuscript snapshots for the N=20 oscillator case are extracted.
All snapshots appearing in the manuscript are included here for reference.


generate_fig2.py: A Python script that reproduces the metastable dynamics and generates the time series plots shown in Figure 2 of the manuscript. 
Model Details: The code simulates a generalized Kuramoto model with non-reciprocal adaptive couplings.
System Size: $N=10$ (adjustable)
Adaptation: Hebbian learning ($\epsilon_1$) for upper triangular couplings and anti-Hebbian learning ($\epsilon_2$) for lower triangular couplings.
Dynamics: Demonstrates the switching between metastable anti-phase synchronized clusters.
Requirements:
Python 3.x
NumPy
Matplotlib
Usage:Run the script directly to perform the simulation and generate the figure
