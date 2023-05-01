import numpy as np
import matplotlib.pyplot as plt
from Double_Pendulum import DoublePendulum

class DoublePendulumTests:
    def __init__(self, pendulum, integrator='euler_cromer', enable_damping=True, enable_external_forces=True):
        self.pendulum = pendulum
        self.integrator = integrator
        self.enable_damping = enable_damping
        self.enable_external_forces = enable_external_forces

    def test_initial_angle(self, pendulum):
        
        theta1 = np.linspace(0, np.pi/2, 10)
        theta2 = self.pendulum.theta2_0
        omega1 = self.pendulum.omega1_0
        omega2 = self.pendulum.omega2_0

        results = []
        for angle in theta1:
            self.pendulum.theta1_0 = angle
            self.pendulum.simulate(integrator=self.integrator, enable_damping=self.enable_damping, enable_external_forces=self.enable_external_forces)
            results.append([self.pendulum.theta2[-1], self.pendulum.omega2[-1]])

        results = np.array(results)
        plt.plot(theta1, results[:, 0], label='Position')
        plt.plot(theta1, results[:, 1], label='Velocity')
        plt.xlabel('Initial angle of first pendulum (radians)')
        plt.legend()
        plt.show()
        
    def test_energy_conservation(self, pendulum):
        
        self.pendulum.simulate(integrator=self.integrator, enable_damping=self.enable_damping, enable_external_forces=self.enable_external_forces)
        t = self.pendulum.t
        E = self.pendulum.energy()
        
        plt.plot(t, E - E[0])
        plt.xlabel('Time (s)')
        plt.ylabel('Energy difference (J)')
        plt.show()

    def test_sensitivity(self, pendulum):
        
        self.pendulum.simulate(integrator=self.integrator, enable_damping=self.enable_damping, enable_external_forces=self.enable_external_forces)
        t = self.pendulum.t
        y = np.array([self.pendulum.theta1, self.pendulum.theta2, self.pendulum.omega1, self.pendulum.omega2]).T

        theta1_perturbed = self.pendulum.theta1_0 + 0.01
        pendulum_perturbed = DoublePendulum(
            m1=self.pendulum.m1, m2=self.pendulum.m2, l1=self.pendulum.l1, l2=self.pendulum.l2, 
            theta1_0=theta1_perturbed, theta2_0=self.pendulum.theta2_0, 
            omega1_0=self.pendulum.omega1_0, omega2_0=self.pendulum.omega2_0, 
            t_max=self.pendulum.t_max, dt=self.pendulum.dt, 
            d1=self.pendulum.d1, d2=self.pendulum.d2
        )
        pendulum_perturbed.simulate()
        y_perturbed = np.array([pendulum_perturbed.theta1, pendulum_perturbed.theta2, pendulum_perturbed.omega1, pendulum_perturbed.omega2]).T

        delta_y = np.abs(y_perturbed - y)
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].plot(t, delta_y[:, 0])
        axs[0, 0].set_title('Position: Theta1')
        axs[0, 1].plot(t, delta_y[:, 1])
        axs[0, 1].set_title('Position: Theta2')
        axs[1, 0].plot(t, delta_y[:, 2])
        axs[1, 0].set_title('Velocity: Omega1')
        axs[1, 1].plot(t, delta_y[:, 3])
        axs[1, 1].set_title('Velocity: Omega2')
        for ax in axs.flat:
            ax.set(xlabel='Time (s)', ylabel='Difference')
        plt.tight_layout()
        plt.show()

    def test_position_velocity(self, pendulum):
        
        theta1 = np.pi / 4
        theta2 = np.pi / 2
        omega1 = 0
        omega2 = 0
        dp = DoublePendulum(
    m1=pendulum.m1, m2=pendulum.m2, l1=pendulum.l1, l2=pendulum.l2,
    theta1_0=theta1, theta2_0=theta2, omega1_0=omega1, omega2_0=omega2,
    t_max=10, dt=0.01,
    d1=pendulum.d1, d2=pendulum.d2
)

        dp.simulate(integrator=self.integrator, enable_damping=self.enable_damping, enable_external_forces=self.enable_external_forces)

        theta1_perturbed = theta1 + 0.01
        dp_perturbed = DoublePendulum(
    m1=pendulum.m1, m2=pendulum.m2, l1=pendulum.l1, l2=pendulum.l2,
    theta1_0=theta1_perturbed, theta2_0=theta2, omega1_0=omega1, omega2_0=omega2,
    t_max=10, dt=0.01,
    d1=pendulum.d1, d2=pendulum.d2
)

        dp_perturbed.simulate(integrator=self.integrator, enable_damping=self.enable_damping, enable_external_forces=self.enable_external_forces)

        # Find the index of the closest time point to t=5
        idx = np.argmin(np.abs(dp.t - 5))

        # Plot the position and velocity of the second bob for both simulations at t=5
        plt.plot(dp.theta2[idx], dp.omega2[idx], 'o', label='Unperturbed')
        plt.plot(dp_perturbed.theta2[idx], dp_perturbed.omega2[idx], 'o', label='Perturbed')
        plt.xlabel('Position of second bob (m)')
        plt.ylabel('Velocity of second bob (m/s)')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    pendulum = DoublePendulum(
        m1=1.0, m2=1.0, l1=1.0, l2=1.0,
        theta1_0=np.pi/4, theta2_0=np.pi/2,
        omega1_0=0.0, omega2_0=0.0,
        t_max=30.0, dt=0.01,
        d1=0.1, d2=0.1
    )
    pendulum_tests = DoublePendulumTests(pendulum, integrator='euler_cromer', enable_damping=False, enable_external_forces=False)
    pendulum_tests.test_initial_angle(pendulum)
    pendulum_tests.test_energy_conservation(pendulum)
    pendulum_tests.test_sensitivity(pendulum)
    pendulum_tests.test_position_velocity(pendulum)