import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class CubicController:
    """
    Clase que implementa el algoritmo de control de congestión TCP Cubic.
    """
    
    def __init__(self, C, beta, dt):
        self.C = C
        self.beta = beta
        self.dt = dt
        self.reset()
    
    def reset(self):
        self.W_max = 0.0
        self.t_epoch = 0.0 
        self.K = 0.0
        self.ssthresh = float('inf') # Slow Start Threshold inicial infinito
        self.in_slow_start = True
    
    def calculate_K(self, W_max):
        if W_max <= 0 or self.C <= 0:
            return 0.0
        return np.cbrt((W_max * self.beta) / self.C)
    
    def update(self, W_current, t_current, packet_loss_detected, RTT):
        """
        Actualiza W(t).
        RTT es necesario para el crecimiento proporcional físico.
        """
        
        # 1. Manejo de Pérdidas (Perturbación)
        if packet_loss_detected:
            self.W_max = W_current
            # Reducción multiplicativa
            W_new = self.W_max * (1 - self.beta)
            
            # Actualizar ssthresh
            self.ssthresh = self.W_max * (1 - self.beta)
            
            # Recalcular K para Cubic
            self.K = self.calculate_K(self.W_max)
            
            # Reiniciar epoch
            self.t_epoch = 0.0
            
            # Salir de Slow Start (si estábamos ahí)
            self.in_slow_start = False
            
            return max(1.0, W_new)
        
        # 2. Slow Start (Crecimiento Lineal Suave)
        if self.in_slow_start:
            if W_current >= self.ssthresh:
                self.in_slow_start = False
                # Transición a Cubic
                pass
            else:
                # Crecimiento lineal muy suave en lugar de exponencial
                # Incremento constante por tiempo
                growth = 1.0 * self.dt / RTT  # Crece 1 paquete por RTT
                return W_current + growth

        # 3. Congestion Avoidance (Cubic + Reno Friendly)
        
        # Crecimiento Cúbico Pure (Sin Reno)
        # W_cubic(t) = C * (t - K)^3 + W_max
        W_cubic = self.C * (self.t_epoch - self.K)**3 + self.W_max
        
        # Avanzamos el tiempo del epoch
        self.t_epoch += self.dt
        
        return max(1.0, W_cubic)

class TCPCubicSimulatorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Simulador TCP Cubic - Control Robusto")
        
        # Frame controles
        control_frame = tk.Frame(self.master, bd=5, relief=tk.RIDGE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Frame gráficos
        plot_frame = tk.Frame(self.master, bd=2, relief=tk.SUNKEN)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(control_frame, text="Parámetros de Simulación", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        tk.Label(control_frame, text="Estado Inicial", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.W_initial = self.create_entry(control_frame, "Ventana Inicial (pkts):", "1.0")
        self.throughput_target = self.create_entry(control_frame, "Throughput Objetivo (Mbps):", "45.0")
        
        tk.Label(control_frame, text="Parámetros TCP Cubic", font=("Helvetica", 12, "bold")).pack(pady=(20, 5))
        self.C = self.create_entry(control_frame, "C (cúbica):", "0.4")
        self.beta = self.create_entry(control_frame, "β (reducción):", "0.7")
        
        tk.Label(control_frame, text="Parámetros de Red", font=("Helvetica", 12, "bold")).pack(pady=(20, 5))
        self.rtt_base = self.create_entry(control_frame, "RTT Base (ms):", "50.0")
        self.rtt_variation = self.create_entry(control_frame, "Variación RTT (ms):", "5.0")
        self.loss_probability = self.create_entry(control_frame, "Prob. Pérdida Aleatoria (%):", "0.0005")
        self.buffer_size = self.create_entry(control_frame, "Tamaño Buffer (pkts):", "200.0")
        
        self.simulation_time = self.create_entry(control_frame, "Tiempo Simulación (s):", "60.0")
        
        tk.Button(control_frame, text="Ejecutar Simulación", command=self.run_simulation, bg="green", fg="white").pack(pady=20, fill=tk.X, ipady=5)
        tk.Button(control_frame, text="Limpiar Gráficos", command=self.clear_plots, bg="red", fg="white").pack(pady=5, fill=tk.X, ipady=5)
        
        # 4 Subplots: Window, Throughput, Error, Buffer
        self.fig, self.ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True, 
                                         gridspec_kw={'height_ratios': [2, 2, 1, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barra de herramientas
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.setup_plots()
        self.current_target = None

    def create_entry(self, parent, label_text, default_value):
        frame = tk.Frame(parent)
        frame.pack(pady=5, fill=tk.X)
        tk.Label(frame, text=label_text, width=25, anchor='w').pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=10)
        entry.pack(side=tk.RIGHT)
        entry.insert(0, default_value)
        return entry
    
    def setup_plots(self):
        self.ax[0].set_title("Evolución de la Ventana de Congestión W(t)")
        self.ax[0].set_ylabel("Ventana (pkts)")
        self.ax[0].grid(True)
        
        self.ax[1].set_title("Throughput (Tasa de Transmisión)")
        self.ax[1].set_ylabel("Throughput (Mbps)")
        self.ax[1].grid(True)
        
        self.ax[2].set_title("Error (Objetivo - Real)")
        self.ax[2].set_ylabel("Error (Mbps)")
        self.ax[2].grid(True)
        
        self.ax[3].set_title("Ocupación del Buffer")
        self.ax[3].set_ylabel("Cola (pkts)")
        self.ax[3].set_xlabel("Tiempo (s)")
        self.ax[3].grid(True)
        
        self.fig.tight_layout()
    
    def clear_plots(self):
        for axis in self.ax:
            axis.cla()
        self.setup_plots()
        self.current_target = None
        self.canvas.draw()
    
    def run_simulation(self):
        try:
            W_initial = float(self.W_initial.get())
            throughput_target = float(self.throughput_target.get())
            C = float(self.C.get())
            beta = float(self.beta.get())
            rtt_base_ms = float(self.rtt_base.get())
            rtt_variation_ms = float(self.rtt_variation.get())
            loss_probability = float(self.loss_probability.get()) / 100.0
            # bandwidth_mbps ahora es igual al target para simular tracking
            bandwidth_mbps = throughput_target 
            buffer_size_pkts = float(self.buffer_size.get())
            sim_time = float(self.simulation_time.get())
        except ValueError:
            messagebox.showerror("Error", "Valores numéricos inválidos.")
            return

        if self.current_target is not None and self.current_target != throughput_target:
            self.clear_plots()
            
        dt = 0.01
        n_points = int(sim_time / dt)
        t = np.linspace(0, sim_time, n_points)
        
        cubic = CubicController(C, beta, dt)
        
        # Inicialización
        cubic.W_max = W_initial 
        cubic.in_slow_start = True 
        
        W = np.zeros(n_points)
        throughput = np.zeros(n_points)
        error = np.zeros(n_points)
        rtt = np.zeros(n_points)
        buffer_occupancy = np.zeros(n_points)
        
        W[0] = W_initial
        rtt[0] = rtt_base_ms / 1000.0
        
        rtt_noise = np.random.normal(0, rtt_variation_ms / 1000.0, n_points)
        rtt_base_sec = rtt_base_ms / 1000.0
        
        # Rastrear pérdidas por tipo
        overflow_loss_t = []
        overflow_loss_W = []
        random_loss_t = []
        random_loss_W = []
        
        current_queue = 0.0
        
        for i in range(1, n_points):
            # 1. Dinámica de la Planta (Red + Buffer)
            
            # RTT = RTT_base + RTT_queue + Noise
            # RTT_queue = Queue / Bandwidth_pkts_per_sec
            packet_size_bits = 12000 
            bandwidth_pkts_sec = (bandwidth_mbps * 1e6) / packet_size_bits
            
            rtt_queue = current_queue / bandwidth_pkts_sec if bandwidth_pkts_sec > 0 else 0
            rtt[i] = max(0.001, rtt_base_sec + rtt_queue + rtt_noise[i])
            
            # Dinámica del Buffer
            # Input Rate = W / RTT
            input_rate = W[i-1] / rtt[i] # pkts/sec
            output_rate = bandwidth_pkts_sec # pkts/sec
            
            # Cambio en cola = (Input - Output) * dt
            current_queue += (input_rate - output_rate) * dt
            current_queue = max(0.0, current_queue)
            
            # Pérdida por Buffer Overflow
            is_buffer_loss = False
            if current_queue > buffer_size_pkts:
                current_queue = buffer_size_pkts
                is_buffer_loss = True
            
            buffer_occupancy[i] = current_queue
            
            # Otras pérdidas
            is_random_loss = (np.random.random() < loss_probability)
            
            # Rastrear por tipo
            if is_buffer_loss:
                overflow_loss_t.append(t[i])
                overflow_loss_W.append(W[i-1])
            
            if is_random_loss:
                random_loss_t.append(t[i])
                random_loss_W.append(W[i-1])
            
            packet_loss = is_buffer_loss or is_random_loss
            
            # 2. Actualización del Controlador
            W[i] = cubic.update(W[i-1], t[i], packet_loss, rtt[i])
            
            # 3. Cálculo de Salidas Reales
            # Throughput real es lo que sale de la red SIN CONTAR los paquetes dropeados
            # Si el buffer no está vacío, salimos a tasa máxima (BW)
            # Si está vacío, salimos a tasa de entrada
            if current_queue > 0:
                real_throughput_pkts = bandwidth_pkts_sec
            else:
                real_throughput_pkts = min(input_rate, bandwidth_pkts_sec)
            
            # Si hubo pérdida de paquetes, el throughput efectivo baja
            # porque los paquetes perdidos NO llegan al destino
            if packet_loss:
                # Cuando hay pérdida, asumimos que una fracción de los paquetes se pierde
                # Esto reduce el throughput efectivo
                # En un modelo simple: si hay overflow, se pierde el exceso
                if is_buffer_loss:
                    # Si hay overflow, los paquetes que exceden el buffer se pierden
                    # Asumimos que el input rate excede el output rate
                    excess_rate = max(0, input_rate - output_rate)
                    real_throughput_pkts = max(0, real_throughput_pkts - excess_rate)
                    
            throughput[i] = (real_throughput_pkts * packet_size_bits) / 1e6
            
            error[i] = throughput_target - throughput[i]
            
        plot_label = f'C={C}, β={beta}'
        if self.current_target is None:
            self.ax[1].axhline(y=throughput_target, color='r', linestyle='--', label=f'Objetivo ({throughput_target} Mbps)')
            self.current_target = throughput_target
            
        self.ax[0].plot(t, W, label=plot_label, linewidth=1.5)
        
        # Graficar pérdidas por tipo con diferentes colores
        if overflow_loss_t:
            self.ax[0].scatter(overflow_loss_t, overflow_loss_W, color='orange', marker='x', s=50, zorder=5, label='Overflow')
        if random_loss_t:
            self.ax[0].scatter(random_loss_t, random_loss_W, color='red', marker='x', s=50, zorder=5, label='Pérdida Aleatoria')
            
        self.ax[1].plot(t, throughput, linewidth=1.5)
        # Mostrar límite de ancho de banda
        self.ax[1].axhline(y=bandwidth_mbps, color='g', linestyle=':', alpha=0.5, label='Ancho de Banda')
        
        self.ax[2].plot(t, error, linewidth=1.5)
        
        self.ax[3].plot(t, buffer_occupancy, color='purple', linewidth=1.5)
        self.ax[3].axhline(y=buffer_size_pkts, color='k', linestyle='--', alpha=0.5, label='Tamaño Buffer')
             
        self.ax[0].legend()
        self.ax[1].legend()
        self.ax[3].legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TCPCubicSimulatorGUI(root)
    root.mainloop()
