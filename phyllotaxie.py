import math as m
import numpy as np

def tableau_des_plastochron_ratio():
    r0 = 0.35
    rho = 0.20
    r_asymptote = 1.03
    try:
        n = int(input("define t: "))
    except:
        n = 50

    tableau_intermediaire = r0 * np.exp(-rho * np.arange(1, n + 1)) + r_asymptote
    
    # Initialization
    r_t0 = 1
    tab_vieillisement = [0] * n
    try:
        tab_vieillisement[0] = 1
        tab_vieillisement[1] = 1
    except:
        pass
    mU = []
    r_t = []
    theta = []
    gamma = 0.2
    r_t1 = r_t0 * tableau_intermediaire[0]
    r_t2 = r_t1 * tableau_intermediaire[1]
    r_t.append(r_t2)
    r_t.append(r_t1)
    theta.append(60.0)
    theta.append(240.0)

    angles = np.arange(0, 3601) / 10.0
    
    t = 0
    while t < n:
        if t == 0:
            age = len(r_t) - 1
        elif t == 1:
            age = len(r_t) - 2
        else:
            age = 0
        
        alpha_rad = np.radians(angles - theta[t])
        valeur = np.sqrt(1 + r_t[t]**2 - 2*r_t[t]*np.cos(alpha_rad))
        U_t = np.exp(-valeur/gamma - age*tab_vieillisement[t])
        
        if t == 0:
            sU = U_t.copy()
        else:
            sU += U_t
        
        if t == len(r_t) - 1:
            # Find minimum index
            min_idx = np.argmin(sU)
            mU.append(sU[min_idx])
            theta.append(min_idx / 10.0)
            
            try:
                len_r_t = len(r_t)
                # Vectorized multiplication
                r_t = [r * tableau_intermediaire[len_r_t] for r in r_t]
                r_t.append(tableau_intermediaire[len_r_t])
                t = -1
            except IndexError:
                pass
            
            if t == n - 1:
                # Calculate divergence angles
                div = [(theta[i + 1] - theta[i]) % 360 for i in range(len(theta) - 1)]
                
                # Extract only whole-number angles (every 10th index: 0°, 1°, 2°, ..., 360°)
                sU_whole_angles = sU[::10]  # 361 values

                from itertools import zip_longest
                with open("thetas_et_r_de_t.csv", "w") as fd:
                    fd.write("t; div; theta r; Rt; Umin; u\n")
                    
                    rows = zip_longest(
                        range(1, len(div) + 1),
                        div, theta, r_t, mU,
                        sU_whole_angles,
                        fillvalue=""
                    )
                    for row in rows:
                        fd.write("; ".join(str(v) for v in row) + "\n")

                print(f"div: {div}")
                print(f"theta: {theta}")
                print(f"r_t: {r_t}")
        
        t += 1

tableau_des_plastochron_ratio()
