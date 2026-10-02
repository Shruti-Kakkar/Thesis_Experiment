import pandas as pd

df = pd.read_csv("/home/student/s/skakkar/scratch/dev-uos/projects/VTCAV_Dermatology/outputs2/bwv_calibration/bwv_calibration_attributions.csv")

present = df.loc[df["bwv_label"] == 1, "attribution"].values
absent = df.loc[df["bwv_label"] == 0, "attribution"].values
n1, n2 = len(present), len(absent)

from scipy import stats
u_stat, p_value = stats.mannwhitneyu(present, absent, alternative="greater")
rank_biserial = (2 * u_stat) / (n1 * n2) - 1  # corrected

print(f"U={u_stat:.2f}, p={p_value:.6g}, rank-biserial={rank_biserial:.3f}")