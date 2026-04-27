# 33-BUS EXPERIMENT BATCH RUNNER
# Run all 6 experiments sequentially (Agent 2 & 3, Seeds 0-2)

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║    33-BUS EXPERIMENTS: 6 RUNS (2 AGENTS × 3 SEEDS)        ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

$startTime = Get-Date

# ============================================================
# AGENT 2 (Unshielded) - 3 Seeds
# ============================================================

Write-Host "`n[1/6] Agent 2 Seed 0 - Unshielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops.py --env-id IEEE33-v0 --epochs 100 --steps-per-epoch 10000 --seed 0 --logdir final_results_33bus/agent2_seed0

Write-Host "`n[2/6] Agent 2 Seed 1 - Unshielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops.py --env-id IEEE33-v0 --epochs 100 --steps-per-epoch 10000 --seed 1 --logdir final_results_33bus/agent2_seed1

Write-Host "`n[3/6] Agent 2 Seed 2 - Unshielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops.py --env-id IEEE33-v0 --epochs 100 --steps-per-epoch 10000 --seed 2 --logdir final_results_33bus/agent2_seed2

# ============================================================
# AGENT 3 (Shielded) - 3 Seeds
# ============================================================

Write-Host "`n[4/6] Agent 3 Seed 0 - Shielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops_hybrid.py --env-id IEEE33-Hybrid-v0 --epochs 100 --steps-per-epoch 10000 --seed 0 --logdir final_results_33bus/agent3_seed0

Write-Host "`n[5/6] Agent 3 Seed 1 - Shielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops_hybrid.py --env-id IEEE33-Hybrid-v0 --epochs 100 --steps-per-epoch 10000 --seed 1 --logdir final_results_33bus/agent3_seed1

Write-Host "`n[6/6] Agent 3 Seed 2 - Shielded 33-bus" -ForegroundColor Cyan
conda activate omnisafe310
python .\launch_focops_hybrid.py --env-id IEEE33-Hybrid-v0 --epochs 100 --steps-per-epoch 10000 --seed 2 --logdir final_results_33bus/agent3_seed2

# ============================================================
# COMPLETION SUMMARY
# ============================================================

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           ALL 33-BUS EXPERIMENTS COMPLETE! ✅              ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Host "Total Time: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Cyan
Write-Host "`nResults saved to: final_results_33bus/`n" -ForegroundColor Yellow

Write-Host "To view results, run:" -ForegroundColor White
Write-Host "  tensorboard --logdir=`"final_results_33bus`" --port=6007`n" -ForegroundColor Cyan
