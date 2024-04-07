import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pyransac3d import Plane
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.
from tkinter import messagebox
def read_coordinates_from_file(file_path):
    root = tk.Tk()
    root.withdraw()  # ukrywamy główne okno tkinter
    coords = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.split()
                if len(values) != 3:
                    raise ValueError("Każdy wiersz powinien zawierać dokładnie trzy wartości liczbowe oddzielone spacjami.")
                coords.append([float(val) for val in values])
    except ValueError as e:
        messagebox.showerror("Błąd", f"Wystąpił błąd: {e}. Plik nie spełnia wymagań: każdy wiersz musi zawierać trzy wartości liczbowe oddzielone spacjami.")
        file_path = choose_file()  # Funkcja do wyboru nowego pliku
        if file_path:
            return read_coordinates_from_file(file_path)  # Rekurencyjne wywołanie w przypadku nowego pliku
        else:
            messagebox.showwarning("Wybór pliku", "Nie wybrano pliku. Kończenie działania programu.")
            root.destroy()
            return None

    root.destroy()
    return np.array(coords)

def choose_file():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Wybierz plik", "Wybierz plik z danymi oddzielonymi spacjami. Możesz skorzystać gotowych: ("
                                        "'cylindrical.xyz, vertical.xyz lub horizontal.xyz')")

    if messagebox.askyesno("Potwierdzenie", "Czy Twój plik spełnia podane warunki?"):
        file_path = filedialog.askopenfilename(title="Wybierz plik.")
        return file_path
    else:
        print("Program wymaga pliku z danymi oddzielonymi spacjami. Proces został anulowany.")
        root.destroy()  # zamykamy okno
        return None



def fit_plane_ransac(coords):
    plane = Plane()
    best_eq, best_inliers = plane.fit(coords, thresh=0.01, minPoints=100, maxIteration=1000)
    return best_eq, best_inliers


def is_plane_horizontal(normal_vector):
    # A plane is considered horizontal if its normal vector has a small X and Y components compared to Z component
    return abs(normal_vector[0]) < 0.1 and abs(normal_vector[1]) < 0.1


def visualize_plane(coords, best_eq):
    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='b')

    # Plot the fitted plane
    point  = np.array([0, 0, best_eq[3]])  # A point on the plane
    normal = np.array(best_eq[:3])  # Normal vector to the plane
    # Create a meshgrid of points
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    if np.isclose(normal[0], 0, atol=1e-6) and np.isclose(normal[2], 0, atol=1e-6):
        Y_const = normal[2] / normal[1]
        X,Z = np.meshgrid(np.linspace(xlim[0], xlim[1],10), np.linspace(zlim[0], zlim[1],10))
        Y = Y_const * np.ones_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.5, color='r')
    else:
        X,Y = np.meshgrid(np.linspace(xlim[0], xlim[1],10), np.linspace(ylim[0], ylim[1],10))
        Z = (-normal[0] * X - normal[1] * Y - best_eq[3]) * 1. /normal[2]
        ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def analyze_plane(coords):
    best_eq, best_inliers = fit_plane_ransac(coords)
    normal_vector = best_eq[:3]
    print("Wektor normalny do płaszczyzny:", normal_vector)

    if len(best_inliers) > 0.8 * len(coords):  # Adjust this threshold based on your needs
        print("Chmura punktów jest płaszczyzną.")
        if is_plane_horizontal(normal_vector):
            print("Chmura punktów jest płaszczyzną horyzontalną.")
        else:
            print("Chmura punktów jest płaszczyzną pionową lub inną.")
        visualize_plane(coords, best_eq)
    else:
        print("Chmura punktów nie jest płaszczyzną lub jest cylindryczna")


def main():
    file_path = choose_file()
    coords = read_coordinates_from_file(file_path)
    analyze_plane(coords)


if __name__ == "__main__":
    main()
