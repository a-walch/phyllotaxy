Requires numpy; the pressure models also require scipy. Each script is run directly and prints its results; no figures are produced.

### phyllotaxie.py

Computes linum phyllotaxis (Walch, JTB 2022; equation 2)  
Input r(n): r0; rho; rasymptote  
try: n: number of primordia  
output for each primordium: divergence angle; trigonometric angle; distance from center  
output from 0 to 360 degrees : inhibition potential for the last primordium  
written to theta_r_and_u.csv

### phyllotaxis_pressure_model.py

Pressure model of the Arabidopsis shoot apical meristem:
physical pressure field
(tunica membrane on a Winkler foundation, forced by point-like primordia,
Green function K0).  
Also used as a module by predict_next_primordia.py.  
output: derived physical parameters (membrane tension, screening length,
maxima and minima of the pressure profile on the central zone periphery)

### winkler_model.py

Forced Winkler block only, with the parameters of the Methods section
(rho = 1.12, gamma = 0.5, contact radius growing from P0 to P6).  
output: derived physical parameters; positions of the generated spiral
P0..P6; the three pressure minima on the central zone periphery, i.e. the
predicted sites of P-1, P-2, P-3, and their deviation from the naive 137.5
degree continuation of the spiral.

### predict_next_primordia.py

Same P-1, P-2, P-3 prediction, built on phyllotaxis_pressure_model.py, with
an organ radius growing linearly with age and a plastochron ratio of 1.12.  
output: generated positions; predicted minima and their deviation from the
naive continuation of the spiral.
