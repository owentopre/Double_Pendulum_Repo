import numpy as np
import matplotlib.pyplot as plt

class DoublePendulum:
    def __init__(self, m1, m2, l1, l2, theta1_0, theta2_0, omega1_0, omega2_0, t_max, dt, d1=0, d2=0):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.theta1_0 = theta1_0
        self.theta2_0 = theta2_0
        self.omega1_0 = omega1_0
        self.omega2_0 = omega2_0
        self.t_max = t_max
        self.dt = dt
        self.d1 = d1
        self.d2 = d2
        self.g = 9.81

        self.theta1 = np.zeros(int(t_max/dt) + 1)
        self.theta2 = np.zeros(int(t_max/dt) + 1)
        self.omega1 = np.zeros(int(t_max/dt) + 1)
        self.omega2 = np.zeros(int(t_max/dt) + 1)

        self.theta1[0] = theta1_0
        self.theta2[0] = theta2_0
        self.omega1[0] = omega1_0
        self.omega2[0] = omega2_0

    def external_force(self, t):
        F_ext = 0.5
        freq = 1.0
        return F_ext * np.sin(2 * np.pi * freq * t)

    def simulate(self, integrator='euler_cromer', enable_damping=True, enable_external_forces=True):
        self.t = np.arange(0, self.t_max + self.dt, self.dt)
        for i in range(len(self.theta1) - 1):
            if integrator == 'euler_cromer':
                self._euler_cromer_step(i, enable_damping, enable_external_forces)
            elif integrator == 'rk4':
                self._rk4_step(i, enable_damping, enable_external_forces)
            else:
                raise ValueError("Invalid integrator. Supported options are 'euler_cromer' and 'rk4'.")
    
    def _euler_cromer_step(self, i, enable_damping, enable_external_forces):
        t = i * self.dt

        F1_ext = self.external_force(t) / self.m1 if enable_external_forces else 0
        F2_ext = self.external_force(t) / self.m2 if enable_external_forces else 0

        damping1 = self.d1 * self.omega1[i] if enable_damping else 0
        damping2 = self.d2 * self.omega2[i] if enable_damping else 0

        alpha1 = (self.m2*self.g*np.sin(self.theta2[i])*np.cos(self.theta1[i]-self.theta2[i]) 
                  - self.m2*self.l2*self.omega2[i]**2*np.sin(self.theta1[i]-self.theta2[i]) 
                  - (self.m1+self.m2)*self.g*np.sin(self.theta1[i]) + F1_ext) / (self.l1*(self.m1+self.m2*np.sin(self.theta1[i]-self.theta2[i])**2)) - damping1
        alpha2 = ((self.m1+self.m2)*(self.l1*self.omega1[i]**2*np.sin(self.theta1[i]-self.theta2[i]) - self.g*np.sin(self.theta2[i]) + F2_ext) 
                  + self.m2*self.l2*self.omega2[i]**2*np.sin(self.theta1[i]-self.theta2[i])*np.cos(self.theta1[i]-self.theta2[i])) / (self.l2*(self.m1+self.m2*np.sin(self.theta1[i]-self.theta2[i])**2)) - damping2

        self.omega1[i+1] = self.omega1[i] + alpha1 * self.dt
        self.omega2[i+1] = self.omega2[i] + alpha2 * self.dt
        self.theta1[i+1] = self.theta1[i] + self.omega1[i+1] * self.dt
        self.theta2[i+1] = self.theta2[i] + self.omega2[i+1] * self.dt
        
    def _rk4_step(self, i, enable_damping, enable_external_forces):
        def derivs(t, theta1, theta2, omega1, omega2):
            F1_ext = self.external_force(t) / self.m1 if enable_external_forces else 0
            F2_ext = self.external_force(t) / self.m2 if enable_external_forces else 0

            damping1 = self.d1 * omega1 if enable_damping else 0
            damping2 = self.d2 * omega2 if enable_damping else 0

            alpha1 = (self.m2*self.g*np.sin(theta2)*np.cos(theta1-theta2) 
                      - self.m2*self.l2*omega2**2*np.sin(theta1-theta2) 
                      - (self.m1+self.m2)*self.g*np.sin(theta1) + F1_ext) / (self.l1*(self.m1+self.m2*np.sin(theta1-theta2)**2)) - damping1
            alpha2 = ((self.m1+self.m2)*(self.l1*omega1**2*np.sin(theta1-theta2) - self.g*np.sin(theta2) + F2_ext) 
                      + self.m2*self.l2*omega2**2*np.sin(theta1-theta2)*np.cos(theta1-theta2)) / (self.l2*(self.m1+self.m2*np.sin(theta1-theta2)**2)) - damping2

            return omega1, omega2, alpha1, alpha2

        dt = self.dt
        t = i * self.dt

        k1_theta1, k1_theta2, k1_omega1, k1_omega2 = derivs(t, self.theta1[i], self.theta2[i], self.omega1[i], self.omega2[i])
        k2_theta1, k2_theta2, k2_omega1, k2_omega2 = derivs(t + 0.5*dt, self.theta1[i] + 0.5*dt*k1_theta1, self.theta2[i] + 0.5*dt*k1_theta2, self.omega1[i] + 0.5*dt*k1_omega1, self.omega2[i] + 0.5*dt*k1_omega2)
        k3_theta1, k3_theta2, k3_omega1, k3_omega2 = derivs(t + 0.5*dt, self.theta1[i] + 0.5*dt*k2_theta1, self.theta2[i] + 0.5*dt*k2_theta2, self.omega1[i] + 0.5*dt*k2_omega1, self.omega2[i] + 0.5*dt*k2_omega2)
        k4_theta1, k4_theta2, k4_omega1, k4_omega2 = derivs(t + dt, self.theta1[i] + dt*k3_theta1, self.theta2[i] + dt*k3_theta2, self.omega1[i] + dt*k3_omega1, self.omega2[i] + dt*k3_omega2)

        self.theta1[i+1] = self.theta1[i] + dt * (k1_theta1 + 2*k2_theta1 + 2*k3_theta1 + k4_theta1) / 6
        self.theta2[i+1] = self.theta2[i] + dt * (k1_theta2 + 2*k2_theta2 + 2*k3_theta2 + k4_theta2) / 6
        self.omega1[i+1] = self.omega1[i] + dt * (k1_omega1 + 2*k2_omega1 + 2*k3_omega1 + k4_omega1) / 6
        self.omega2[i+1] = self.omega2[i] + dt * (k1_omega2 + 2*k2_omega2 + 2*k3_omega2 + k4_omega2) / 6

    def energy(self):
        """Calculate the total energy of the system (potential + kinetic)"""
        g = 9.81
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        theta1, theta2 = self.theta1, self.theta2
        omega1, omega2 = self.omega1, self.omega2

        # Calculate the potential energy
        pe = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)

        # Calculate the kinetic energy
        ke = 0.5 * m1 * (l1 * omega1)**2 + 0.5 * m2 * ((l1 * omega1)**2 + (l2 * omega2)**2 + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))

        # Return the total energy
        return pe + ke

    def plot_simulation(self):
        x1 = self.l1 * np.sin(self.theta1)
        y1 = -self.l1 * np.cos(self.theta1)
        x2 = x1 + self.l2 * np.sin(self.theta2)
        y2 = y1 - self.l2 * np.cos(self.theta2)

        plt.plot(x1, y1, label='Bob 1')
        plt.plot(x2, y2, label='Bob 2')
        plt.xlabel('Horizontal position (m)')
        plt.ylabel('Vertical position (m)')
        plt.title('Double Pendulum Simulation')
        plt.legend()
        plt.show()

# Default initial conditions
pendulum = DoublePendulum(
    m1=1.0, m2=1.0, l1=1.0, l2=1.0, 
    theta1_0=np.pi/4, theta2_0=np.pi/2, 
    omega1_0=0.0, omega2_0=0.0, 
    t_max=30.0, dt=0.01, 
    d1=0.1, d2=0.1
)
pendulum.simulate(integrator='euler_cromer', enable_damping=True, enable_external_forces=True)
pendulum.plot_simulation()