import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tkinter import messagebox


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
        file_path = filedialog.askopenfilename(title="Wybierz plik")
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
        # Wybierz losowe trzy punkty
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
def ask_user_for_fitting():
    response = input("Czy dopasować płaszczyznę do wykrytej chmury cylindrycznej? (tak/nie): ")
    return response.strip().lower() == 'tak'

def main():
    global distances, x_cluster, y_cluster, z_cluster
    file_path = choose_file()
    if not file_path:
        print("Nie wybrano pliku.")
        return
    try:
        x_coords, y_coords, z_coords = read_coordinates_from_file(file_path)
        points = np.column_stack((x_coords, y_coords, z_coords))
        kmeans = KMeans(n_clusters=3)
        labels = kmeans.fit_predict(points)
        colors = ['r', 'g', 'b']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        user_decision_to_fit_plane = None
        for i in range(3):
            cluster_points = points[labels == i]
            x_cluster, y_cluster, z_cluster = cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2]
            if (np.all(x_cluster) and np.all(y_cluster) and np.all(z_cluster)):
                print(f"Chmura {i}: Wykryto płaszczyznę cylindryczną.")
                if user_decision_to_fit_plane is None:  # Pytamy użytkownika tylko jeśli jeszcze nie podjął decyzji
                    user_decision_to_fit_plane = ask_user_for_fitting()
                if not user_decision_to_fit_plane:
                    print("Pomijanie dopasowania płaszczyzny dla wszystkich chmur. Działanie programu zakonczone")
                    return 0  # Jeśli użytkownik nie chce dopasowywać płaszczyzn, przerywamy pętlę
            best_plane, inliers = fit_plane_ransac(x_cluster, y_cluster, z_cluster)
            if best_plane is not None:
                A, B, C, D = best_plane
                normal_vector = np.array(best_plane[:3])
                print(f"Chmura {i}: Wektor normalny do płaszczyzny: {normal_vector}")
                plane_eq = lambda p: A * p[0] + B * p[1] + C * p[2] + D
                avg_distance = np.mean(distances)
                print(f"Chmura {i}: Średnia odległość do płaszczyzny: {avg_distance}")
                if avg_distance == 0:
                    print(f"Chmura {i} jest uznana za płaszczyznę.")
                    if np.isclose(A, 0, atol=1e-6) and np.isclose(B, 0, atol=1e-6):
                        print("Płaszczyzna pozioma\n")
                    elif np.isclose(A, 0, atol=1e-6) and np.isclose(C, 0, atol=1e-6):
                        print("Płaszczyzna pionowa równoległa do osi z.\n")
                    else:
                        print("Płaszczyzna ogólna.\n")
                else:
                    print(f"Chmura {i} nie jest uznana za płaszczyznę.\n")
                if np.isclose(A, 0, atol=1e-6) and np.isclose(C, 0, atol=1e-6):
                    # Płaszczyzna pionowa równoległa do osi Z (Y=constant), A = 0 i C = 0
                    Y_const = -D / B
                    xx, zz = np.meshgrid(np.linspace(x_cluster.min(), x_cluster.max(), 10),
                                         np.linspace(z_cluster.min(), z_cluster.max(), 10))
                    yy = Y_const * np.ones_like(xx)
                    ax.plot_surface(xx, yy, zz, alpha=0.5, color=colors[i], label='Płaszczyzna pionowa: Y=const')
                else:
                    xx, yy = np.meshgrid(np.linspace(x_cluster.min(), x_cluster.max(), 10),
                                         np.linspace(y_cluster.min(), y_cluster.max(), 10))
                    zz = (-D - A * xx - B * yy) / C
                    ax.plot_surface(xx, yy, zz, alpha=0.5, color=colors[i])
            ax.scatter(x_cluster, y_cluster, z_cluster, color=colors[i], marker='o', label=f'Chmura {i}')
            ax.set_title('Chmury punktów 3D z klasteryzacji k-średnich i dopasowanymi płaszczyznami')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    except FileNotFoundError:
        print("Plik nie został znaleziony.")


if __name__ == "__main__":
    main()

