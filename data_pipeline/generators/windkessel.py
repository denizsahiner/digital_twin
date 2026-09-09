import numpy as np

def generate_inflow(t, hr, sv, systole_duration):
    """
    Kalpten çıkan kan akışını (Q_in) simüle eder.
    Sistolde yarım sinüs, Diyastolde 0.
    """
    cycle_duration = 60.0 / hr
    local_t = t % cycle_duration

    if local_t < systole_duration:
        q_max = (sv * np.pi) / (2.0 * systole_duration)
        return q_max * np.sin(np.pi * local_t / systole_duration)
    else:
        return 0.0


def wk3_derivative(p, t, r_total, c_total, z_c, hr, sv, systole_duration):
    """
    Windkessel 3 yaklaşımı
    """
    r_peripheral = r_total - z_c
    if r_peripheral <= 1e-6:
        r_peripheral = 1e-6

    q_in = generate_inflow(t, hr, sv, systole_duration)
    dp_dt = (q_in / c_total) - (p / (r_peripheral * c_total))
    return dp_dt
