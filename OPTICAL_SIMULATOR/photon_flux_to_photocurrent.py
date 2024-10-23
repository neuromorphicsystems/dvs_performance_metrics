# From the optical simulator
Phi_density = 1e18   # Photon flux density (photons/m²·s)

# Given parameters - Constants
q   = 1.602e-19  # Elementary charge (C)
p   = 10e-6            # Pixel pitch (m)
eta = 0.5            # Quantum efficiency

# Step 1: Calculate Pixel Area
A_pd = p ** 2
print(f"Pixel area A_pd = {A_pd:.3e} m²")

# Step 2: Calculate Photon Flux
Phi = Phi_density * A_pd
print(f"Photon flux Phi = {Phi:.3e} photons/s")

# Step 3: Compute Photocurrent
I_ph = eta * q * Phi
print(f"Photocurrent I_ph = {I_ph:.3e} A")
