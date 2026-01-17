# scenarios.py

def get_scenario_config(scenario_name):
    """
    Senaryo adına göre Hedef Çarpanları ve Değişim Hızını döndürür.
    target_mults: {Parametre: Çarpan}
    rate: Değişim Hızı (0.01 yavaş, 0.50 çok hızlı)
    lifestyle_flags: ML modeli için {'smoke': 1, 'alco': 0} gibi bayraklar
    """

    # Varsayılan (Etkisiz)
    config = {
        'target_mults': {'R': 1.0, 'C': 1.0, 'HR': 1.0, 'SV': 1.0, 'Zc': 1.0},
        'rate': 0.02,
        'lifestyle_flags': {}  # Değişiklik yok
    }

    if scenario_name == "sigara_icmek":
        config['target_mults'] = {'R': 1.25, 'C': 0.80, 'HR': 1.10, 'SV': 0.95, 'Zc': 1.10}
        config['rate'] = 0.005  # Yavaş kötüleşme
        config['lifestyle_flags'] = {'smoke': 1}

    elif scenario_name == "sigarayi_birakmak":
        # Genetik tabana dönüş (hafif kalıcı hasarla)
        config['target_mults'] = {'R': 1.05, 'C': 0.95, 'HR': 1.0, 'SV': 1.0, 'Zc': 1.0}
        config['rate'] = 0.01  # Yavaş iyileşme
        config['lifestyle_flags'] = {'smoke': 0}

    elif scenario_name == "spor_yapmak":
        config['target_mults'] = {'R': 0.85, 'C': 1.15, 'HR': 0.80, 'SV': 1.20, 'Zc': 1.0}
        config['rate'] = 0.008
        config['lifestyle_flags'] = {'active': 1}  # Eğer modelde active varsa

    elif scenario_name == "sedanter_yasam":
        config['target_mults'] = {'R': 1.05, 'C': 1.0, 'HR': 1.10, 'SV': 0.90, 'Zc': 1.0}
        config['rate'] = 0.005
        config['lifestyle_flags'] = {'active': 0}

    elif scenario_name == "alkol_tuketmek":
        config['target_mults'] = {'R': 0.90, 'C': 1.0, 'HR': 1.15, 'SV': 1.0, 'Zc': 1.0}
        config['rate'] = 0.50  # Hızlı etki
        config['lifestyle_flags'] = {'alco': 1}

    elif scenario_name == "grip_olmak":
        config['target_mults'] = {'R': 0.85, 'C': 1.0, 'HR': 1.25, 'SV': 1.0, 'Zc': 1.0}
        config['rate'] = 0.20
        # Grip yaşam tarzı değildir, bayrak değiştirmez

    elif scenario_name == "istirahat":
        # YENİ SENARYO: Tam İyileşme (Fabrika ayarlarına/Genetiğe hızlı dönüş)
        config['target_mults'] = {'R': 1.0, 'C': 1.0, 'HR': 1.0, 'SV': 1.0, 'Zc': 1.0}
        config['rate'] = 0.10 # İyileşme hızı
        config['lifestyle_flags'] = {'active': 0}

    return config