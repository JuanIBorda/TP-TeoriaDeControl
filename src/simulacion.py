import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class CubicController:
    """
    Implementación de TCP Cubic (como la tenías).
    """

    def __init__(self, C, beta, dt):
        self.C = C
        self.beta = beta
        self.dt = dt
        self.reset()

    def reset(self):
        self.W_max = 0.0
        self.t_last_loss = 0.0
        self.in_congestion_avoidance = True

    def calculate_K(self, W_max):
        if W_max <= 0 or self.C <= 0:
            return 0.0
        return np.cbrt(W_max * (1 - self.beta) / self.C)

    def cubic_value(self, W_current, t_current):
        if self.W_max <= 0:
            return W_current
        K = self.calculate_K(self.W_max)
        t_since_loss = t_current - self.t_last_loss
        return self.C * (t_since_loss - K) ** 3 + self.W_max

    def standard_value(self, W_current):
        # la función estándar (Reno-like) para comparar internamente
        # (la lógica original usaba esto; no lo mostramos)
        return W_current + 1.0 / W_current

    def update(self, W_current, t_current, packet_loss_detected):
        # comportamiento ante pérdida
        if packet_loss_detected:
            self.W_max = W_current
            self.t_last_loss = t_current
            return max(1.0, self.beta * W_current)

        # modo congestion avoidance (CUBIC)
        if self.in_congestion_avoidance and self.W_max > 0:
            W_cubic = self.cubic_value(W_current, t_current)
            W_standard = self.standard_value(W_current)
            return max(W_cubic, W_standard)

        return W_current + 1.0


class TCPCubicSimulatorGUI:
    """
    Simulador TCP Cubic con modelo de planta (enlace con capacidad y buffer).
    Mantiene zoom/pan (NavigationToolbar) intactos.
    """

    def __init__(self, master):
        self.master = master
        self.master.title("Simulador TCP Cubic - Planta con Buffer y Capacidad")

        control_frame = tk.Frame(self.master, bd=5, relief=tk.RIDGE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plot_frame = tk.Frame(self.master, bd=2, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(control_frame, text="Parámetros de Simulación",
                 font=("Helvetica", 14, "bold")).pack(pady=10)

        tk.Label(control_frame, text="Estado Inicial",
                 font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.W_initial = self.create_entry(control_frame, "Ventana Inicial (pkts):", "10.0")
        self.throughput_target = self.create_entry(control_frame, "Throughput Objetivo (Mbps):", "10.0")

        tk.Label(control_frame, text="Parámetros TCP Cubic",
                 font=("Helvetica", 12, "bold")).pack(pady=(20, 5))
        self.C = self.create_entry(control_frame, "C (cúbica):", "0.4")
        self.beta = self.create_entry(control_frame, "β (reducción):", "0.7")

        tk.Label(control_frame, text="Parámetros de Red (Planta)",
                 font=("Helvetica", 12, "bold")).pack(pady=(20, 5))
        self.rtt_base = self.create_entry(control_frame, "RTT Base (ms):", "50.0")
        self.rtt_variation = self.create_entry(control_frame, "Variación RTT (ms):", "5.0")
        self.loss_probability = self.create_entry(control_frame, "Prob. Pérdida Aleatoria (%):", "0.5")

        self.bandwidth_limit = self.create_entry(control_frame, "Ancho de Banda (Mbps):", "50.0")
        self.buffer_size = self.create_entry(control_frame, "Buffer (paquetes):", "200")

        self.simulation_time = self.create_entry(control_frame, "Tiempo Simulación (s):", "60.0")

        tk.Button(control_frame, text="Ejecutar Simulación",
                  command=self.run_simulation, bg="green", fg="white").pack(
            pady=15, fill=tk.X, ipady=5)

        tk.Button(control_frame, text="Limpiar Gráficos",
                  command=self.clear_plots, bg="red", fg="white").pack(
            pady=5, fill=tk.X, ipady=5)

        # --- FIGURA: 4 subplots (W, throughput, error, buffer occupancy) ---
        self.fig, self.ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 1, 1, 2]})

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        self.setup_plots()
        self.current_target = None

    def create_entry(self, parent, label_text, default_value):
        frame = tk.Frame(parent)
        frame.pack(pady=4, fill=tk.X)
        tk.Label(frame, text=label_text, width=22, anchor='w').pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=12)
        entry.insert(0, default_value)
        entry.pack(side=tk.RIGHT)
        return entry

    def setup_plots(self):
        titles = [
            "Ventana de Congestión W(t)",
            "Throughput (Mbps)",
            "Error (Mbps)",
            "Ocupación de Buffer (paquetes)"
        ]
        ylabels = [
            "Ventana (pkts)",
            "Mbps",
            "Mbps",
            "Paquetes"
        ]

        for a, t, y in zip(self.ax, titles, ylabels):
            a.grid(True, linestyle='--', alpha=0.5)
            a.set_title(t)
            a.set_ylabel(y)

        self.ax[-1].set_xlabel("Tiempo (s)")
        self.fig.tight_layout()

    def clear_plots(self):
        for axis in self.ax:
            axis.cla()
        self.setup_plots()
        self.current_target = None
        self.canvas.draw()

    def run_simulation(self):
        # --- leer parámetros GUI ---
        try:
            W_initial = float(self.W_initial.get())
            throughput_target = float(self.throughput_target.get())
            C = float(self.C.get())
            beta = float(self.beta.get())
            rtt_base_ms = float(self.rtt_base.get())
            rtt_variation_ms = float(self.rtt_variation.get())
            loss_probability = float(self.loss_probability.get()) / 100.0
            bandwidth_limit_mbps = float(self.bandwidth_limit.get())
            buffer_size_packets = float(self.buffer_size.get())
            sim_time = float(self.simulation_time.get())
        except Exception as e:
            messagebox.showerror("Error", f"Valores inválidos: {e}")
            return

        # --- constantes de simulación ---
        dt = 0.01
        n_points = int(max(2, sim_time / dt))
        t = np.linspace(0, sim_time, n_points)

        packet_size_bits = 12000.0  # como en tu versión original (12k bits por paquete)

        # service_rate en paquetes/segundo (capacidad del enlace en packets/sec)
        service_rate_pkts_per_sec = (bandwidth_limit_mbps * 1e6) / packet_size_bits

        # ruido RTT
        noise = np.random.normal(0, rtt_variation_ms / 1000.0, n_points)
        base_rtt = rtt_base_ms / 1000.0

        # --- inicializar controlador y vectores ---
        cubic = CubicController(C, beta, dt)
        cubic.W_max = W_initial

        W = np.zeros(n_points)
        throughput = np.zeros(n_points)
        error = np.zeros(n_points)
        rtt = np.zeros(n_points)
        queue = np.zeros(n_points)  # ocupación de buffer (paquetes)
        packet_loss_events = np.zeros(n_points, dtype=bool)

        W[0] = W_initial
        rtt[0] = max(0.0005, base_rtt + noise[0])  # evitar rtt 0

        # simulación temporal
        for i in range(1, n_points):
            # --- tasa de llegada que genera el emisor según la ventana y el RTT anterior
            # arrival_rate en paquetes/segundo (approx.)
            arrival_rate_pkts = 0.0
            if rtt[i - 1] > 0:
                arrival_rate_pkts = W[i - 1] / rtt[i - 1]

            # actualizar cola/ buffer (simpliificado: cola += (λ - μ) * dt)
            delta_queue = (arrival_rate_pkts - service_rate_pkts_per_sec) * dt
            queue[i] = max(0.0, queue[i - 1] + delta_queue)

            # determinar delay por cola (segundos) y RTT dinámico
            queue_delay = 0.0
            if service_rate_pkts_per_sec > 0:
                queue_delay = queue[i] / service_rate_pkts_per_sec
            rtt[i] = max(0.0005, base_rtt + noise[i] + queue_delay)

            # detectar pérdida por saturación de buffer
            overflow_loss = False
            if queue[i] >= buffer_size_packets:
                overflow_loss = True
                queue[i] = buffer_size_packets  # saturado

            # pérdida aleatoria (ruido, errores de enlace) + pérdida por overflow
            random_loss = (np.random.random() < loss_probability)
            packet_loss = random_loss or overflow_loss

            if packet_loss:
                packet_loss_events[i] = True

            # actualizar controlador (usa packet_loss detectado)
            W_new = cubic.update(W[i - 1], t[i], packet_loss)
            W[i] = W_new

            # throughput efectivo: lo que la planta puede entregar = min(arrival_rate, service_rate) * packet_size_bits
            delivered_pkts_per_sec = min(arrival_rate_pkts, service_rate_pkts_per_sec)
            throughput[i] = (delivered_pkts_per_sec * packet_size_bits) / 1e6  # Mbps

            error[i] = throughput_target - throughput[i]

        # --- plotear resultados manteniendo zoom/pan ---
        # W(t)
        self.ax[0].plot(t, W, label="W(t)", linewidth=1.2)
        if np.any(packet_loss_events):
            self.ax[0].scatter(t[packet_loss_events], W[packet_loss_events], color='red', marker='x', label="Pérdida (overflow/aleatoria)")
        self.ax[0].legend()

        # Throughput (con línea de objetivo)
        self.ax[1].plot(t, throughput, linewidth=1.2)
        if self.current_target is None:
            self.ax[1].axhline(throughput_target, color='r', linestyle='--', label="Objetivo")
            self.current_target = throughput_target
        self.ax[1].legend()

        # Error
        self.ax[2].plot(t, error, linewidth=1.0)

        # Buffer occupancy
        self.ax[3].plot(t, queue, linewidth=1.2)
        # marcar el tamaño del buffer
        self.ax[3].axhline(buffer_size_packets, color='r', linestyle='--', alpha=0.7, label="Tamaño Buffer")
        self.ax[3].legend()

        self.ax[0].set_ylabel("Ventana (pkts)")
        self.ax[1].set_ylabel("Mbps")
        self.ax[2].set_ylabel("Mbps")
        self.ax[3].set_ylabel("Paquetes")
        self.ax[3].set_xlabel("Tiempo (s)")

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = TCPCubicSimulatorGUI(root)
    root.mainloop()
