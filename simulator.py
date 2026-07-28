import numpy as np
import pandas as pd
import warnings
from scenarios import get_scenario_config

MAX_SIMULATION_DAYS = 365


class BioDigitalTwin:
    def __init__(self, r, c, zc, hr, sv, active_conditions=None):
        if active_conditions is None:
            active_conditions = []

        self.current_params = {'R': r, 'C': c, 'Zc': zc, 'HR': hr, 'SV': sv}
        self.active_conditions = active_conditions
        self.history = []
        self.total_days = 0

        self.current_lifestyle = {'smoke': 0, 'alco': 0}
        if 'sigara' in active_conditions:
            self.current_lifestyle['smoke'] = 1
        if 'alkol' in active_conditions:
            self.current_lifestyle['alco'] = 1

        self.genetic_baseline = self._reverse_engineer_genetics()

    def _reverse_engineer_genetics(self):
        base = self.current_params.copy()
        if 'sigara' in self.active_conditions:
            base['R'] /= 1.20
            base['C'] /= 0.85
        if 'sedanter' in self.active_conditions:
            base['SV'] /= 0.90
            base['HR'] /= 1.10
        return base

    def run_simulation(self, days, scenario_name, description=""):
        if self.total_days + days > MAX_SIMULATION_DAYS:
            allowed_days = max(0, MAX_SIMULATION_DAYS - self.total_days)
            warnings.warn(
                f"Simulation horizon capped at max 1 year (365 days). "
                f"Requested: {days} days, allowed: {allowed_days} days."
            )
            days = allowed_days

        if days <= 0:
            print("Simulation reached maximum 1-year boundary.")
            return

        config = get_scenario_config(scenario_name)
        target_mults = config['target_mults']
        rate = config['rate']

        self.current_lifestyle.update(config['lifestyle_flags'])

        target_values = {}
        for param, mult in target_mults.items():
            target_values[param] = self.genetic_baseline[param] * mult

        print(f"Running {days} days for scenario '{scenario_name}' ({description}) [Progress: {self.total_days + days}/365 days]")

        for _ in range(days):
            self.total_days += 1

            aging_factor = np.exp(-0.008 / 365.0)
            self.current_params['C'] *= aging_factor

            for param in ['R', 'C', 'Zc', 'HR', 'SV']:
                curr = self.current_params[param]
                targ = target_values[param]
                self.current_params[param] = curr + rate * (targ - curr)

            bp_sys, bp_dia = self.get_current_bp()
            self.history.append({
                'Day': self.total_days,
                'Scenario': description,
                'R': self.current_params['R'],
                'C': self.current_params['C'],
                'HR': self.current_params['HR'],
                'SBP': bp_sys,
                'DBP': bp_dia,
                'Smoke_Flag': self.current_lifestyle['smoke']
            })

    def get_current_bp(self):
        co = (self.current_params['SV'] * self.current_params['HR']) / 60.0
        mean_p = self.current_params['R'] * co
        pulse_p = self.current_params['SV'] / (self.current_params['C'] + 1e-9)
        return mean_p + (pulse_p * 2.0 / 3.0), mean_p - (pulse_p / 3.0)