# Double_Pendulum_Repo

Double_Pendulum.py 
This script contains the DoublePendulum, which, in turn, contains the main simulation loops as methods (euler-cromer and rk4) as well as functions that add an external force to the simulation and calculate the total energy of the system at any given point

Double_Pendulum_Test.py
This script contains 4 testing methods within the DoublePendulumTests class:
test_initial_angle: Runs the simulation for a variety of different starting angles for the first pendulum then graphs the corresponding positions and velocities of the second bob (at a fixed time)
test_energy_conservation: Plots the energy of the simulation over time
test_sensitivity: runs a normal simulation and one with a slight differency in the initial angle of the first pendulum then graphs the difference of the positions and velocities of the masses (the difference between the peturbed and unpeturbed simulated masses) over time
test_position_velocity: Similar to the previous simulation but plots a single point ie. the position and velocity of the second masses (peturbed and unpeturbed) at a point in time

There are 3 main factors to change in the simulation other than the physical dimensions of the apparatus and initial conditions:
Integrator: can be rk4 or euler-cromer
enalge_damping: Boolean controling whether damping forces are included in the simulation
enable_external_forces: boolean controlling the inclusion of the external_force method (this method could be edited to approximate an exernal force to the system)
