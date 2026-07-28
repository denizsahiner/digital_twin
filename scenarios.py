def get_scenario_config(scenario_name):
    config = {
        'target_mults': {'R': 1.0, 'C': 1.0, 'HR': 1.0, 'SV': 1.0, 'Zc': 1.0},
        'rate': 0.02,
        'lifestyle_flags': {}
    }

    if scenario_name == "sigara_icmek":
        config['target_mults'] = {'R': 1.25, 'C': 0.80, 'HR': 1.10, 'SV': 0.95, 'Zc': 1.10}
        config['rate'] = 0.005
        config['lifestyle_flags'] = {'smoke': 1}

    elif scenario_name == "sigarayi_birakmak":
        config['target_mults'] = {'R': 1.05, 'C': 0.95, 'HR': 1.00, 'SV': 1.00, 'Zc': 1.00}
        config['rate'] = 0.010
        config['lifestyle_flags'] = {'smoke': 0}

    elif scenario_name == "spor_yapmak":
        config['target_mults'] = {'R': 0.85, 'C': 1.15, 'HR': 0.80, 'SV': 1.20, 'Zc': 1.00}
        config['rate'] = 0.008
        config['lifestyle_flags'] = {'active': 1}

    elif scenario_name == "sedanter_yasam":
        config['target_mults'] = {'R': 1.05, 'C': 1.00, 'HR': 1.10, 'SV': 0.90, 'Zc': 1.00}
        config['rate'] = 0.005
        config['lifestyle_flags'] = {'active': 0}

    elif scenario_name == "alkol_tuketmek":
        config['target_mults'] = {'R': 0.90, 'C': 1.00, 'HR': 1.15, 'SV': 1.00, 'Zc': 1.00}
        config['rate'] = 0.500
        config['lifestyle_flags'] = {'alco': 1}

    elif scenario_name == "grip_olmak":
        config['target_mults'] = {'R': 0.85, 'C': 1.00, 'HR': 1.25, 'SV': 1.00, 'Zc': 1.00}
        config['rate'] = 0.200
        config['lifestyle_flags'] = {}

    elif scenario_name == "istirahat":
        config['target_mults'] = {'R': 1.00, 'C': 1.00, 'HR': 1.00, 'SV': 1.00, 'Zc': 1.00}
        config['rate'] = 0.100
        config['lifestyle_flags'] = {'active': 0}

    return config