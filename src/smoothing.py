import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the One Euro Filter."""
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.alpha = self._alpha(self.min_cutoff)

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * 30.0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev
        
        if t_e <= 0.0:
            return self.x_prev

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

class MultiChannelSmoother:
    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        """
        Manages filters for multiple channels (bones).
        
        Args:
            min_cutoff: Min cutoff frequency (lower = smoother but more lag).
                        1.0 is good for slow movements.
            beta: Speed coefficient (higher = less lag on fast movement).
                  0.0 = constant cutoff. 0.5-1.0 is usually good for mocap.
        """
        self.filters = {}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.start_time = time.time()

    def update(self, data_dict):
        """
        Update filters with new data dictionary.
        
        Args:
            data_dict: Dictionary {key: [v1, v2, ...]} (e.g., bone rotations/positions)
            
        Returns:
            New dictionary with smoothed values.
        """
        current_time = time.time()
        smoothed_data = {}
        
        for key, value in data_dict.items():
            if not isinstance(value, (list, tuple)):
                smoothed_data[key] = value
                continue
                
            if key not in self.filters:
                self.filters[key] = [
                    OneEuroFilter(current_time, v, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                    for v in value
                ]
                smoothed_data[key] = value
            else:
                filters = self.filters[key]
                
                if len(filters) != len(value):
                    self.filters[key] = [
                        OneEuroFilter(current_time, v, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                        for v in value
                    ]
                    smoothed_data[key] = value
                    continue
                
                new_values = []
                for i, v in enumerate(value):
                    new_values.append(filters[i](current_time, v))
                
                smoothed_data[key] = new_values
                
        return smoothed_data