import matplotlib.pyplot as plt
import pandas as pd


def plot_simulation_dashboard(df_history):
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    color_bp = '#d62728'
    color_hr = '#2ca02c'
    color_res = '#9467bd'
    color_health = '#1f77b4'

    axes[0].fill_between(df_history['Day'], df_history['DBP'], df_history['SBP'], color='red', alpha=0.1)
    axes[0].plot(df_history['Day'], df_history['SBP'], color=color_bp, label='Systolic BP')
    axes[0].plot(df_history['Day'], df_history['DBP'], color='orange', label='Diastolic BP')
    axes[0].set_title('Blood Pressure Contour (mmHg)')
    axes[0].legend()

    axes[1].plot(df_history['Day'], df_history['HR'], color=color_hr, lw=2)
    axes[1].set_title('Heart Rate (bpm)')

    axes[2].plot(df_history['Day'], df_history['R'], color=color_res, lw=2)
    axes[2].set_title('Vascular Resistance (R)')

    ax4 = axes[3]
    health = df_history['Heart_Health_Score']
    smooth_score = health.rolling(window=7, min_periods=1).mean()
    ax4.plot(df_history['Day'], health, color=color_health, alpha=0.3)
    ax4.plot(df_history['Day'], smooth_score, color=color_health, lw=3, label='Health Score Trend')
    ax4.axhspan(80, 100, color='green', alpha=0.1)
    ax4.axhspan(50, 80, color='yellow', alpha=0.1)
    ax4.axhspan(0, 50, color='red', alpha=0.1)
    ax4.set_ylim(0, 105)
    ax4.set_title('Heart Health Score (0-100)')

    change_points = df_history[df_history['Scenario'] != df_history['Scenario'].shift(1)]
    for idx, row in change_points.iterrows():
        for ax in axes:
            ax.axvline(x=row['Day'], color='black', ls=':', alpha=0.4)
        axes[0].text(row['Day'], df_history['SBP'].max() * 1.05, str(row['Scenario']), rotation=90, fontsize=8)

    plt.tight_layout()
    plt.show()