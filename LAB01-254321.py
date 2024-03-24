import numpy as np
from tkinter import filedialog
import tkinter as tk


def horizontal_surface(width, length, num_points):

    x_coords = np.random.uniform(low=-width / 2, high=width / 2, size=num_points)
    y_coords = np.random.uniform(low=-length / 2, high=length / 2, size=num_points)

    z_coords = np.zeros(num_points)

    point_cloud = np.column_stack((x_coords, y_coords, z_coords))

    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".xyz",
                                             filetypes=(("XYZ files", "*.xyz"), ("All files", "*.*")),
                                             initialfile=f"point_cloud_horizontal_{num_points}.xyz")

    if save_path:
        np.savetxt(save_path, point_cloud, delimiter=' ', fmt='%.6f')
        print("Point cloud saved successfully.")
    else:
        print("Save operation canceled.")


def vertical_surface(width, height, num_points):

    x_coords = np.random.uniform(low=-width / 2, high=width / 2, size=num_points)
    z_coords = np.random.uniform(low=0, high=height, size=num_points)

    y_coords = np.zeros(num_points)

    point_cloud = np.column_stack((x_coords, y_coords, z_coords))

    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".xyz",
                                             filetypes=(("XYZ files", "*.xyz"), ("All files", "*.*")),
                                             initialfile=f"point_cloud_vertical_{num_points}.xyz")
    if save_path:
        np.savetxt(save_path, point_cloud, delimiter=' ', fmt='%.6f')
        print("Point cloud saved successfully.")
    else:
        print("Save operation canceled.")


def cylindrical_surface(radius, height, num_points):
    theta_coords = np.random.uniform(low=0, high=2 * np.pi, size=num_points)
    z_coords = np.random.uniform(low=0, high=height, size=num_points)

    x_coords = radius * np.cos(theta_coords)
    y_coords = radius * np.sin(theta_coords)

    point_cloud = np.column_stack((x_coords, y_coords, z_coords))

    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".xyz",
                                             filetypes=(("XYZ files", "*.xyz"), ("All files", "*.*")),
                                             initialfile=f"point_cloud_cylindrical_{num_points}.xyz")
    if save_path:
        np.savetxt(save_path, point_cloud, delimiter=' ', fmt='%.6f')
        print("Point cloud saved successfully.")
    else:
        print("Save operation canceled.")

#cylindrical
radius = 5
height = 10
num_points = 1000
cylindrical_surface(radius, height, num_points)

#horizontal
width = 10
length = 10
num_points = 1000
horizontal_surface(width, length, num_points)

#vertical
width = 10
height = 5
num_points = 1000
vertical_surface(width, height, num_points)
