# 33-Bus Experiment Runner
# Run this script to execute all 6 experiments

Write-Host "`n=====================================" -ForegroundColor Green
Write-Host "33-BUS EXPERIMENTS: 6 RUNS STARTING" -ForegroundColor Green
Write-Host "=====================================`n" -ForegroundColor Green

$startTime = Get-Date

# Agent 2 Seed 0
Write-Host "[1/6] Agent 2 Seed 0 - Unshielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops.py --env-id IEEE33-v0 --epochs 100 --steps-per-epoch 10000 --seed 0 --logdir final_results_33bus/agent2_seed0

# Agent 2 Seed 1
Write-Host "`n[2/6] Agent 2 Seed 1 - Unshielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops.py --env-id IEEE33-v0 --epochs 100 --steps-per-epoch 10000 --seed 1 --logdir final_results_33bus/agent2_seed1

# Agent 2 Seed 2
Write-Host "`n[3/6] Agent 2 Seed 2 - Unshielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops.py --env-id IEEE33-v0 --epochs 100 --steps-per-epoch 10000 --seed 2 --logdir final_results_33bus/agent2_seed2

# Agent 3 Seed 0
Write-Host "`n[4/6] Agent 3 Seed 0 - Shielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops_hybrid.py --env-id IEEE33-Hybrid-v0 --epochs 100 --steps-per-epoch 10000 --seed 0 --logdir final_results_33bus/agent3_seed0

# Agent 3 Seed 1
Write-Host "`n[5/6] Agent 3 Seed 1 - Shielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops_hybrid.py --env-id IEEE33-Hybrid-v0 --epochs 100 --steps-per-epoch 10000 --seed 1 --logdir final_results_33bus/agent3_seed1

# Agent 3 Seed 2
Write-Host "`n[6/6] Agent 3 Seed 2 - Shielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops_hybrid.py --env-id IEEE33-Hybrid-v0 --epochs 100 --steps-per-epoch 10000 --seed 2 --logdir final_results_33bus/agent3_seed2

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n=====================================" -ForegroundColor Green
Write-Host "ALL 33-BUS EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Total Time: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Cyan
Write-Host "Results: final_results_33bus/" -ForegroundColor Yellow
Write-Host "View: tensorboard --logdir=`"final_results_33bus`" --port=6007`n" -ForegroundColor Cyan
