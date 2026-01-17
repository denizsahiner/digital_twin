import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BioDigitalTwin:
    def __init__(self, r, c, zc, hr, sv, active_conditions=None):
        """
        r, c, ... : Hastanın ŞU ANKİ (Modelden tahmin edilen) değerleri.
        active_conditions : Liste. Hastanın halihazırda sahip olduğu durumlar.
                            Örn: ['sigara', 'sedanter'] (Hareketsiz)
        """
        if active_conditions is None:
            active_conditions = []

        self.current_params = {'R': r, 'C': c, 'Zc': zc, 'HR': hr, 'SV': sv}
        self.active_conditions = active_conditions
        self.history = []
        self.total_days = 0

        # --- KRİTİK ADIM: GENETİK POTANSİYELİ HESAPLA ---
        # Hastanın "Hiçbir kötü alışkanlığı olmasaydı" değerleri ne olurdu?
        # Bunu buluyoruz ki "Bırakma" senaryosunda nereye iyileşeceğini bilelim.
        self.genetic_baseline = self._reverse_engineer_genetics()

    def _reverse_engineer_genetics(self):
        """
        Mevcut hasarları geri alarak 'İdeal/Genetik' tabanı tahmin eder.
        """
        base = self.current_params.copy()

        # Eğer hasta sigara içiyorsa, Genetik R'si şimdikinden düşüktür.
        if 'sigara' in self.active_conditions:
            # Sigara R'yi %20 artırır, C'yi %15 azaltır varsayımıyla ters işlem:
            base['R'] = base['R'] / 1.20
            base['C'] = base['C'] / 0.85

        # Eğer hasta spor yapmıyorsa (sedanter), Genetik SV'si daha yüksek olabilir
        if 'sedanter' in self.active_conditions:
            # Hareketsizlik SV'yi %10 düşürür, HR'yi %10 artırır varsayımı:
            base['SV'] = base['SV'] / 0.90
            base['HR'] = base['HR'] / 1.10

        # Eğer hasta alkol alıyorsa
        if 'alkol' in self.active_conditions:
            base['HR'] = base['HR'] / 1.05  # Alkol nabzı artırmıştı, geri al

        return base

    def _get_target_state(self, scenario):
        """
        Senaryoya göre vücudun gitmek istediği YENİ hedefi belirler.
        Referans noktası her zaman GENETİK TABAN'dır.
        """
        # Önce Genetik Tabanı al (Temiz sayfa)
        target = self.genetic_baseline.copy()

        # Senaryonun etkilerini Genetik Taban üzerine ekle
        # NOT: 'scenario' o an yapılan eylemdir.

        # --- SENARYO MANTIĞI ---

        if scenario == "sigara_icmek":
            # Genetik tabanın üzerine hasar ekle
            target['R'] *= 1.25  # Kronik içicilik %25 direnç artışı
            target['C'] *= 0.80  # %20 esneklik kaybı
            target['Zc'] *= 1.10

        elif scenario == "sigarayi_birakmak":
            # Hedef doğrudan Genetik Taban'dır (Hasar çarpanı yok)
            # Ancak tam iyileşme asla %100 olmaz, biraz kalıcı hasar bırakalım:
            target['R'] *= 1.05  # %5 kalıcı hasar
            target['C'] *= 0.95

        elif scenario == "spor_yapmak":
            # Genetik tabandan daha iyiye git
            target['R'] *= 0.85  # Damarlar genişler
            target['C'] *= 1.15  # Esneklik artar
            target['HR'] *= 0.85  # Atlet kalbi
            target['SV'] *= 1.20  # Güçlü kalp

        elif scenario == "sedanter_yasam":  # Sporu bırakmak / Hareketsizlik
            target['R'] *= 1.05
            target['HR'] *= 1.10
            target['SV'] *= 0.90

        elif scenario == "grip_olmak":
            target['R'] *= 0.70  # Akut vazodilatasyon
            target['HR'] *= 1.40  # Ateşle nabız artışı

        # Eğer senaryo 'mevcut_durumu_koru' ise (None)
        # Hedef, şu anki active_condition'lara göre belirlenir.

        return target

    def run_simulation(self, days, scenario, description=""):
        """
        Simülasyon motoru (Üstel Yumuşatma ile zaman geçişi)
        """
        # Hız Faktörleri (Değişim ne kadar sürede olur?)
        # 0.01 = Yavaş (Aylar), 0.50 = Hızlı (Günler)
        rates = {
            'sigara_icmek': 0.005,  # Kötüleşme yavaştır
            'sigarayi_birakmak': 0.01,  # İyileşme yavaştır
            'spor_yapmak': 0.008,  # Kondisyon tutmak zaman alır
            'grip_olmak': 0.40,  # Grip bir günde vurur
            'iyilesme_grip': 0.20  # Gripten çıkış hızlıdır
        }

        # Varsayılan hız
        rate = rates.get(scenario, 0.02)

        # Hedef durumu belirle
        target_state = self._get_target_state(scenario)

        print(f"\n⏳ {days} Günlük Süreç: {description} ({scenario})")

        for _ in range(days):
            self.total_days += 1

            for param in ['R', 'C', 'Zc', 'HR', 'SV']:
                curr = self.current_params[param]
                targ = target_state[param]

                # Exponential Smoothing: Yeni = Eski + Hız * (Hedef - Eski)
                self.current_params[param] = curr + rate * (targ - curr)

            # Kayıt
            bp_sys, bp_dia = self._calc_bp()
            self.history.append({
                'Day': self.total_days,
                'Scenario': description,  # Grafikte göstermek için etiket
                'R': self.current_params['R'],
                'HR': self.current_params['HR'],
                'SBP': bp_sys,
                'DBP': bp_dia
            })

    def _calc_bp(self):
        # Basit BP tahmini
        co = (self.current_params['SV'] * self.current_params['HR']) / 60
        mean_p = self.current_params['R'] * co
        pulse_p = self.current_params['SV'] / self.current_params['C']
        return mean_p + (pulse_p * 2 / 3), mean_p - (pulse_p / 3)

    def plot(self):
        df = pd.DataFrame(self.history)
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 1. Damar Direnci
        ax[0].plot(df['Day'], df['R'], color='firebrick', lw=2)
        ax[0].set_title('Damar Direnci (R) Değişimi')
        ax[0].set_ylabel('R (mmHg.s/mL)')
        ax[0].grid(True, alpha=0.3)

        # 2. Tansiyon
        ax[1].plot(df['Day'], df['SBP'], label='Sistolik', color='purple')
        ax[1].plot(df['Day'], df['DBP'], label='Diyastolik', color='orange')
        ax[1].set_title('Tansiyon Değişimi')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        # Senaryo Bölgeleri
        changes = df['Scenario'].ne(df['Scenario'].shift()).cumsum()
        for idx in changes.drop_duplicates().index:
            if idx > 0:
                plt.axvline(x=df.iloc[idx]['Day'], color='k', ls=':', alpha=0.5)
                ax[0].text(df.iloc[idx]['Day'] + 1, df['R'].min(), df.iloc[idx]['Scenario'], rotation=90)

        plt.tight_layout()
        plt.show()


# --- SENARYO TESTİ ---
if __name__ == "__main__":
    # HASTA PROFİLİ: Ahmet Bey (MIMIC'ten gelen veri)
    # Zaten Sigara İçiyor ve Hareketsiz.
    # Bu yüzden R'si yüksek (1.4), C'si düşük (1.1).
    ahmet_bey = BioDigitalTwin(
        r=1.4, c=1.1, zc=0.08, hr=80, sv=70,
        active_conditions=['sigara', 'sedanter']  # <-- MEVCUT DURUMU
    )

    print(f"Genetik Taban (Sigara/Hareketsizlik olmasaydı R): {ahmet_bey.genetic_baseline['R']:.2f}")

    # 1. İlk 30 gün: Hayatına aynen devam ediyor (Sigara içmeye devam)
    ahmet_bey.run_simulation(30, "sigara_icmek", "Mevcut Yaşam (Sigara)")

    # 2. Sonraki 60 gün: Sigarayı Bırakıyor (İyileşme Başlar)
    ahmet_bey.run_simulation(60, "sigarayi_birakmak", "Sigarayı Bıraktı")

    # 3. Sonraki 60 gün: Spora Başlıyor (Hızlanmış İyileşme)
    ahmet_bey.run_simulation(60, "spor_yapmak", "Spora Başladı")

    # 4. Şanssızlık: Grip Oluyor (10 Gün)
    ahmet_bey.run_simulation(10, "grip_olmak", "Ağır Grip")

    # 5. İyileşme ve Spora Devam (30 Gün)
    ahmet_bey.run_simulation(30, "spor_yapmak", "İyileşme + Spor")

    ahmet_bey.plot()