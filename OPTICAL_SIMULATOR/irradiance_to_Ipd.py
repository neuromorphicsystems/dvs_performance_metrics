'''
Convert irradiance (w/m^2) to photocurrent (A)
'''

E = 50            # Irradiance (W/m²) - FROM THE SCENE GENERATOR

tau_d = 1e-5        # Time constant at dark current - user defined constant
I_dark = 5.5e-15    # Dark current (A)
tau_sf = 2.5e3      # Time constant of the source follower (s) 

h = 6.626e-34       # Planck’s constant (J·s)
c = 3e8             # Speed of light (m/s)
q = 1.602e-19       # Electron charge (C)

p = 10e-6           # Pixel pitch (m) - from prophesee gen4
eta = 0.6           # Quantum Efficiency - from prophesee gen4
lambda_ = 550e-9    # Wavelength (m)

# Step 1: Calculate Photodiode Area A_pd - pixel_pitch^2
A_pd = p ** 2
print(f"Photodiode area A_pd = {A_pd:.3e} m²")

# Step 2: Compute Optical Power P_opt
P_opt = E * A_pd
print(f"Optical power P_opt = {P_opt:.3e} W")

# Step 3: Calculate Responsivity R
R = eta * (q * lambda_) / (h * c)
print(f"Responsivity R = {R:.4f} A/W")

# Step 4: Estimate Photocurrent I
I = R * P_opt
print(f"Photocurrent I = {I:.3e} A")

# Step 5: Calculate the time constant
tau_pr = tau_d * (I_dark/(I * I_dark))
print(f"Time constant tau_pr = {tau_pr:.3e} s")

print(f"Time constant tau_sf = {tau_sf:.3e} s")


if tau_sf > tau_pr:
    print(f"if (tau_sf > tau_pr) - Use time constant of the photo receptor: tau_pr = {tau_pr:3e} s")
else:
    print(f"if (tau_sf < tau_pr) Use time constant of the source follower: tau_sf = {tau_sf:3e} s")
    