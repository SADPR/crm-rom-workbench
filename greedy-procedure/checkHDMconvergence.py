import numpy as np
import matplotlib.pyplot as plt

HDM_start = 1
HDM_end = 32
foldername = 'InitialHDMruns/'

iteration = [-1] * (HDM_end - HDM_start + 1)
residual = [-1] * (HDM_end - HDM_start + 1)
cfl = [-1] * (HDM_end - HDM_start + 1)
elapsed_time = [-1] * (HDM_end - HDM_start + 1)
coords = [-1] * (HDM_end - HDM_start + 1)

# Read coordinates from SampledPointsOutput.txt
sampled_points_file = 'GreedyRuns/SampledPointsOutput.txt'
with open(sampled_points_file, 'r') as f:
    for line in f:
        # Each line format: 001: [6.00e-01, -2.00e+00, 2.00e-01, 0.00e+00, 9.00e-02]
        parts = line.split(':')
        if len(parts) >= 2:
            run_idx = int(parts[0].strip())
            if HDM_start <= run_idx <= HDM_end:
                coord_str = parts[1].strip().strip('[]')
                coords[run_idx - HDM_start] = [float(x) for x in coord_str.split(',')]

for i in range(HDM_start, HDM_end + 1):
    log_file_name = foldername + 'HDMrun{:03d}/log2'.format(i)
    with open(log_file_name, 'r') as file:
        data = file.readlines()
        # Find the last line that looks like It  #: Res = #, Cfl = #, Elapsed Time = # s
        for line in reversed(data):
            if "It" in line and "Res" in line and "Cfl" in line and "Elapsed Time" in line:
                parts = line.split(',')
                iteration[i - HDM_start] = int(parts[0].split(':')[0].strip().split(' ')[-1])
                residual[i - HDM_start] = float(parts[0].split('=')[1].strip())
                print("\n HDMrun{:03d}".format(i), flush=True)
                print("Iteration: {}, Residual: {}".format(iteration[i - HDM_start], residual[i - HDM_start]), flush=True)
                print(coords[i - HDM_start], flush=True)
                if(residual[i - HDM_start] > 1):
                    print('HDMrun{:03d} Solution diverged'.format(i), flush=True)
                    print(coords[i - HDM_start], flush=True)
                break

# Plot the residuals
plt.figure()
plt.semilogy(range(HDM_start, HDM_end + 1), residual, marker='o')
plt.axhline(y=5e-7, color='r', linestyle=':')
plt.axvline(x=16, color='r', linestyle='-')
plt.text(8, 1e-6, r'$M_\infty=0.6$', fontsize=17, color='r', va='bottom', ha='center')
plt.text(24, 1e-6, r'$M_\infty=0.8$', fontsize=17, color='r', va='bottom', ha='center')
plt.xlabel('HDMrun index')
plt.ylabel('Residual')
plt.title('HDM Convergence')
plt.xlim(HDM_start, HDM_end)
plt.xticks(np.linspace(HDM_start, HDM_end, 9))
plt.grid()
plt.tight_layout()
figure_name = foldername[:-1] + '_HDM_convergence.png'
plt.savefig(figure_name, dpi=500)
plt.close()