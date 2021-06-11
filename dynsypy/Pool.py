class Pool:

    def __init__(self, dt, t_end, t0=0, adaptive_step=True):

        self.t = t0
        self.dt = dt
        self.t_end = t_end

        self.systems = []

        self.adaptive_step = adaptive_step

    def add(self, system):
        """adds the system to the self.systems list

        NOTES:
              This list contains systems that must be simulated simultaneously.
        """

        self.systems.append(system)

    def simulate(self):

        while self.t < self.t_end:
            self.t = self.t + self.dt

            if self.t > self.t_end:
                self.t = self.t_end

            for system in self.systems:
                if self.adaptive_step:
                    system.adaptive_step(self.t)
                else:
                    system.step(self.t)
