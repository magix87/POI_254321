import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from tkinter import messagebox

def read_coordinates_from_file(file_path):
    x_coords = []
    y_coords = []
    z_coords = []
    while True:
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    values = line.split()
                    if len(values) != 3:
                        raise ValueError("Plik powinien zawierać dokładnie trzy kolumny danych.")
                    x_coords.append(float(values[0]))
                    y_coords.append(float(values[1]))
                    z_coords.append(float(values[2]))
            break  # Jeśli nie wystąpi błąd, wyjście z pętli while
        except ValueError as e:
            messagebox.showerror("Błąd", f"Wystąpił błąd: {e}. Plik nie spełnia wymagań programu - powinien zawierać trzy kolumny z danymi numerycznymi oddzielonymi spacją i skończoną liczbę wierszy.")
            file_path = choose_file()  # Zapytaj o nowy plik
            if not file_path:
                messagebox.showwarning("Wybór pliku", "Nie wybrano pliku. Kończenie działania programu.")
                return None, None, None  # Zwraca None, jeśli użytkownik anuluje wybór pliku

    return np.array(x_coords), np.array(y_coords), np.array(z_coords)


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

def fit_plane_ransac(x, y, z, n_iterations=1000, threshold=0.1):
    global distances
    best_plane = None
    best_inliers = None
    max_inliers = 0

    for _ in range(n_iterations):

        indices = np.random.choice(len(x), 3, replace=False)
        points = np.vstack([x[indices], y[indices], z[indices]]).T

        p1, p2, p3 = points
        normal_vector = np.cross(p2 - p1, p3 - p1)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
        d = -np.dot(normal_vector, p1)

        distances = np.abs(np.dot(points, normal_vector) + d) / np.linalg.norm(normal_vector)

        inliers = np.where(distances < threshold)[0]

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_plane = normal_vector.tolist() + [d]

    return best_plane, best_inliers


def main():
    global dbscan
    file_path = choose_file()
    if not file_path:
        print("Nie wybrano pliku.")
        return
    try:
        x_coords, y_coords, z_coords = read_coordinates_from_file(file_path)
        best_plane, inliers = fit_plane_ransac(x_coords, y_coords, z_coords)
        colors = ['r', 'g', 'b']
        points = np.column_stack((x_coords, y_coords, z_coords))
        kmeans = KMeans(n_clusters=3)
        labels = kmeans.fit_predict(points)
        fig = plt.figure()
        ax2 = fig.add_subplot(121, projection='3d')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        for i in range(3):
            cluster_points = points[labels == i]
            x_cluster, y_cluster, z_cluster = cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]
            ax2.scatter(x_cluster, y_cluster, z_cluster, color=colors[i], marker='o', label=f'Chmura {i}')
            ax2.set_title('Kmeas')
        A, B, C, D = best_plane
        normal_vector = np.array(best_plane[:3])
        print(f"Chmura : Wektor normalny do płaszczyzny: {normal_vector}")
        plane_eq = lambda p: A * p[0] + B * p[1] + C * p[2] + D
        avg_distance = np.mean(distances)
        print(f"Chmura : Średnia odległość do płaszczyzny: {avg_distance}")
        if avg_distance == 0:
            print(f"Chmura  jest uznana za płaszczyznę.")
            if np.isclose(A, 0, atol=1e-6) and np.isclose(B, 0, atol=1e-6):
                print("Płaszczyzna pozioma\n")
                dbscan = DBSCAN(eps=1.7, min_samples=100)  # eps 1.2 dla vertical
            elif np.isclose(A, 0, atol=1e-6) and np.isclose(C, 0, atol=1e-6):
                print("Płaszczyzna pionowa równoległa do osi z.\n")
                dbscan = DBSCAN(eps=1.2, min_samples=100)  # eps 1.2 dla vertical
            else:
                print("Płaszczyzna ogólna.\n")
                dbscan = DBSCAN(eps=3, min_samples=100)  # eps 1.2 dla vertical
        else:
            print(f"Płaszczyzna ogólna lub cylinder.\n")
            dbscan = DBSCAN(eps=3, min_samples=100)  # eps 1.2 dla vertical

        labels = dbscan.fit_predict(points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Liczba znalezionych klastrów: {n_clusters}")
        print(f"Liczba punktów szumowych: {n_noise}")
        ax1 = fig.add_subplot(122, projection='3d')
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xyz = points[class_member_mask]
            ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], marker='o')
        ax1.set_title('3D DBSCAN')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        plt.show()

    except FileNotFoundError:
        print("Plik nie został znaleziony.")



if __name__ == "__main__":
    main()