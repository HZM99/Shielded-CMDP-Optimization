"""
Debug script to understand where violations come from in IEEE69BusEnv
"""
import numpy as np
import torch
from rl_constrained_smartgrid_control.environments.bus69_environment import IEEE69BusEnv

# Create environment
env = IEEE69BusEnv()
state, _ = env.reset(seed=42)

# Take a few random actions and analyze violations
np.random.seed(42)
for step in range(10):
    # Generate random action within bounds
    action = env.action_space.sample()
    
    # Step environment
    next_state, reward, done, _, info = env.step(action)
    
    ineq_viol = info['ineq_viol']
    eq_viol = info['eq_viol']
    
    total_ineq = np.sum(ineq_viol) if isinstance(ineq_viol, np.ndarray) else ineq_viol
    total_eq = np.sum(eq_viol) if isinstance(eq_viol, np.ndarray) else eq_viol
    
    if total_ineq > 0.1:
        print(f"\nStep {step}: Inequality Violation = {total_ineq:.4f}")
        print(f"   Equality Violation = {total_eq:.4f}")
        
        # Analyze which constraints are violated
        if isinstance(ineq_viol, np.ndarray):
            # Grid constraints: 4*ng + 2*nbus
            nineq_grid = 4 * env.ng + 2 * env.nbus
            grid_viols = ineq_viol[:nineq_grid]
            ev_viols = ineq_viol[nineq_grid:]
            
            print(f"   Grid violations: {np.sum(grid_viols):.4f}")
            print(f"   EV violations: {np.sum(ev_viols):.4f}")
            
            # Breakdown by type
            pg_over = ineq_viol[0:env.ng]
            pg_under = ineq_viol[env.ng:2*env.ng]
            qg_over = ineq_viol[2*env.ng:3*env.ng]
            qg_under = ineq_viol[3*env.ng:4*env.ng]
            vm_over = ineq_viol[4*env.ng:4*env.ng+env.nbus]
            vm_under = ineq_viol[4*env.ng+env.nbus:4*env.ng+2*env.nbus]
            
            print(f"   Pg violations: over={np.sum(pg_over):.4f}, under={np.sum(pg_under):.4f}")
            print(f"   Qg violations: over={np.sum(qg_over):.4f}, under={np.sum(qg_under):.4f}")
            print(f"   Vm violations: over={np.sum(vm_over):.4f}, under={np.sum(vm_under):.4f}")
            
            # Check action bounds
            pg_start = env.pg_start_yidx
            qg_start = env.qg_start_yidx
            vm_start = env.vm_start_yidx
            va_start = env.va_start_yidx
            
            pg_action = action[pg_start:qg_start]
            qg_action = action[qg_start:vm_start]
            vm_action = action[vm_start:va_start]
            
            pmax = env.pmax.cpu().numpy()
            pmin = env.pmin.cpu().numpy()
            qmax = env.qmax.cpu().numpy()
            qmin = env.qmin.cpu().numpy()
            vmax = env.vmax.cpu().numpy()
            vmin = env.vmin.cpu().numpy()
            
            print(f"   Pg range: {pg_action} vs [{pmin}, {pmax}]")
            print(f"   Qg range: {qg_action} vs [{qmin}, {qmax}]")
            print(f"   Vm range: min={np.min(vm_action):.4f}, max={np.max(vm_action):.4f} vs [{np.min(vmin):.4f}, {np.max(vmax):.4f}]")
            
            # Direct calculation of violations like environment does
            pg_over_calc = pg_action - pmax
            pg_under_calc = pmin - pg_action
            print(f"   Pg over calculation: {pg_over_calc} (should match {pg_over})")
            print(f"   Pg under calculation: {pg_under_calc} (should match {pg_under})")
            
            # Check if there's a mismatch
            if not np.allclose(pg_over[0], max(0, pg_over_calc[0])):
                print(f"   WARNING: MISMATCH in Pg over violation!")
    
    state = next_state
    if done:
        break

print("\nDebug complete")
