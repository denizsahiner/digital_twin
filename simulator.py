# simulator.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scenarios import get_scenario_config


class BioDigitalTwin:
    def __init__(self, r, c, zc, hr, sv, active_conditions=None):
        if active_conditions is None:
            active_conditions = []

        self.current_params = {'R': r, 'C': c, 'Zc': zc, 'HR': hr, 'SV': sv}
        self.active_conditions = active_conditions
        self.history = []
        self.total_days = 0

        # ML modeli için anlık yaşam tarzı durumu
        self.current_lifestyle = {'smoke': 0, 'alco': 0}
        # Başlangıç koşullarına göre flagleri set et
        if 'sigara' in active_conditions: self.current_lifestyle['smoke'] = 1
        if 'alkol' in active_conditions: self.current_lifestyle['alco'] = 1

        self.genetic_baseline = self._reverse_engineer_genetics()

    def _reverse_engineer_genetics(self):
        """Mevcut hasarı geri alarak genetiği bul"""
        base = self.current_params.copy()
        if 'sigara' in self.active_conditions:
            base['R'] /= 1.20
            base['C'] /= 0.85
        if 'sedanter' in self.active_conditions:
            base['SV'] /= 0.90
            base['HR'] /= 1.10
        return base

    def run_simulation(self, days, scenario_name, description=""):
        """Senaryoyu çalıştırır ve parametreleri günceller."""

        # 1. Senaryo Ayarlarını Çek
        config = get_scenario_config(scenario_name)
        target_mults = config['target_mults']
        rate = config['rate']

        # 2. Lifestyle Flaglerini Güncelle (ML için)
        self.current_lifestyle.update(config['lifestyle_flags'])

        # 3. Hedef Değerleri Hesapla (Genetic Base * Multiplier)
        target_values = {}
        for param, mult in target_mults.items():
            target_values[param] = self.genetic_baseline[param] * mult

        print(f"⏳ {days} Gün: {description} ({scenario_name})")

        # 4. Günlük Döngü
        for _ in range(days):
            self.total_days += 1

            # Exponential Smoothing
            for param in ['R', 'C', 'Zc', 'HR', 'SV']:
                curr = self.current_params[param]
                targ = target_values[param]
                self.current_params[param] = curr + rate * (targ - curr)

            # Kayıt
            bp_sys, bp_dia = self.get_current_bp()
            self.history.append({
                'Day': self.total_days,
                'Scenario': description,
                'R': self.current_params['R'],
                'HR': self.current_params['HR'],
                'SBP': bp_sys,
                'DBP': bp_dia,
                'Smoke_Flag': self.current_lifestyle['smoke']
            })

    def get_current_bp(self):
        """Mevcut parametrelerle Tansiyonu hesaplar"""
        co = (self.current_params['SV'] * self.current_params['HR']) / 60.0
        mean_p = self.current_params['R'] * co
        pulse_p = self.current_params['SV'] / self.current_params['C']
        return mean_p + (pulse_p * 2 / 3), mean_p - (pulse_p / 3)

    def plot(self):
        # (Grafik kodu aynı kalabilir, buraya eklemiyorum yer kaplamasın diye)
        pass