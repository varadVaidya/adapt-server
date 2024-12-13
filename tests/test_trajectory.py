import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t')
A, B, C = 1.25, 1.25, 0.25
a, b, delta = 2/3, 3/3, sp.pi/4
c = sp.lcm(a, b) 
print(c)

x = A * sp.sin(a * t + delta)
y = B * sp.sin(b * t)
z = C * sp.sin(c * t)
vx = sp.diff(x, t)
vy = sp.diff(y, t)
vz = sp.diff(z, t)

ax = sp.diff(vx, t)
ay = sp.diff(vy, t)
az = sp.diff(vz, t)

print("Velocity (x):", vx)
print("Velocity (y):", vy)
print("Velocity (z):", vz)

print("Acceleration (x):", ax)
print("Acceleration (y):", ay)
print("Acceleration (z):", az)

# T_total = 2 * np.pi / np.gcd(a, b)
# print(f'Total Period: {T_total}')
T_total = 20
t_vals = np.linspace(0, T_total * 2, 1000)

x_func = sp.lambdify(t, x, "numpy")
y_func = sp.lambdify(t, y, "numpy")
z_func = sp.lambdify(t, z, "numpy")

vx_func = sp.lambdify(t, vx, "numpy")
vy_func = sp.lambdify(t, vy, "numpy")
vz_func = sp.lambdify(t, vz, "numpy")

ax_func = sp.lambdify(t, ax, "numpy")
ay_func = sp.lambdify(t, ay, "numpy")
az_func = sp.lambdify(t, az, "numpy")

x_vals = x_func(t_vals)
y_vals = y_func(t_vals)
z_vals = z_func(t_vals) + 1.5

vx_vals = vx_func(t_vals)
vy_vals = vy_func(t_vals)
vz_vals = vz_func(t_vals)

ax_vals = ax_func(t_vals)
ay_vals = ay_func(t_vals)
az_vals = az_func(t_vals)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label='Lissajous Curve')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

fig, axs = plt.subplots(3, 2, figsize=(15, 10))

axs[0, 0].plot(t_vals, vx_vals, label='Velocity (X)', color='r')
axs[1, 0].plot(t_vals, vy_vals, label='Velocity (Y)', color='g')
axs[2, 0].plot(t_vals, vz_vals, label='Velocity (Z)', color='b')

axs[0, 1].plot(t_vals, ax_vals, label='Acceleration (X)', color='r')
axs[1, 1].plot(t_vals, ay_vals, label='Acceleration (Y)', color='g')
axs[2, 1].plot(t_vals, az_vals, label='Acceleration (Z)', color='b')

for i, ax in enumerate(axs[:, 0]):
    ax.set_title(f'Velocity Plot {["X", "Y", "Z"][i]}')
    ax.legend()
    ax.grid(True)

for i, ax in enumerate(axs[:, 1]):
    ax.set_title(f'Acceleration Plot {["X", "Y", "Z"][i]}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()