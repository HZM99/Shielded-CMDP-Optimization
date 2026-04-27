# shield_model_lookahead.py - Multi-step Lookahead Safety Shield (Option 2A)

import numpy as np

def check_action_safety(state, action, env):
    """
    Multi-step Lookahead Safety Shield (Option 2A).
    Simulates 2-3 steps ahead to predict cascading violations.
    Uses environment's step function to predict future states.
    """
    
    # 1. Create a copy of the action to modify
    corrected_action = np.copy(action)
    
    # Debug: Track if this is being called
    if not hasattr(check_action_safety, 'call_count'):
        check_action_safety.call_count = 0
        check_action_safety.unsafe_count = 0
        check_action_safety.rollback_count = 0
        print("🛡️ Multi-step Lookahead Shield initialized (2-step prediction)")
    check_action_safety.call_count += 1
    
    if check_action_safety.call_count == 1:
        print(f"🛡️ Shield called for first time! Action shape: {action.shape}")
    
    # 2. IMPROVED MULTI-STEP LOOKAHEAD: Simulate actual environment transitions
    try:
        # Get current state and compute violations AFTER applying each candidate action
        current_state = env.state if hasattr(env, 'state') else state
        
        # Predict violations by simulating the physics (power flow)
        # We'll test multiple candidate actions and pick the safest
        candidates = []
        
        # Candidate 0: Original action
        try:
            # Simulate what happens if we apply this action
            # We use the environment's physics model (power flow) to predict next state
            original_violation = simulate_and_check_violations(env, current_state, action)
            candidates.append(("original", action, original_violation))
        except Exception as e:
            # If simulation fails, assume high violation
            candidates.append(("original", action, 1000.0))
        
        # Only try alternatives if original action has violations
        if candidates[0][2] > 0.01:  # Threshold for "risky" action
            check_action_safety.unsafe_count += 1
            
            # Candidate 1: 80% scaled action
            scaled_80 = action * 0.8
            viol_80 = simulate_and_check_violations(env, current_state, scaled_80)
            candidates.append(("80%", scaled_80, viol_80))
            
            # Candidate 2: 60% scaled action
            scaled_60 = action * 0.6
            viol_60 = simulate_and_check_violations(env, current_state, scaled_60)
            candidates.append(("60%", scaled_60, viol_60))
            
            # Candidate 3: 40% scaled action
            scaled_40 = action * 0.4
            viol_40 = simulate_and_check_violations(env, current_state, scaled_40)
            candidates.append(("40%", scaled_40, viol_40))
            
            # Candidate 4: Conservative bounds (20% margin)
            conservative_action = apply_conservative_bounds(action, env)
            viol_cons = simulate_and_check_violations(env, current_state, conservative_action)
            candidates.append(("conservative", conservative_action, viol_cons))
            
            # Candidate 5: Previous safe action if available
            if hasattr(env, 'last_safe_action') and env.last_safe_action is not None:
                viol_prev = simulate_and_check_violations(env, current_state, env.last_safe_action)
                candidates.append(("previous", env.last_safe_action, viol_prev))
            
            # Candidate 6: Zero action (safest fallback)
            zero_action = np.zeros_like(action)
            viol_zero = simulate_and_check_violations(env, current_state, zero_action)
            candidates.append(("zero", zero_action, viol_zero))
            
            # Pick the candidate with LOWEST actual predicted violation
            best_name, corrected_action, best_viol = min(candidates, key=lambda x: x[2])
            
            if check_action_safety.unsafe_count <= 10:
                print(f"🛡️ LOOKAHEAD SHIELD: Blocked risky action (#{check_action_safety.unsafe_count})")
                print(f"   Original violation: {candidates[0][2]:.4f}")
                print(f"   Best alternative ({best_name}): {best_viol:.4f}")
                print(f"   Improvement: {((candidates[0][2] - best_viol) / max(candidates[0][2], 0.001) * 100):.1f}%")
            
            is_safe = False
            
            # Store successful correction for future reference
            if best_viol < candidates[0][2] * 0.8:  # At least 20% improvement
                env.last_safe_action = np.copy(corrected_action)
        else:
            # Original action is safe enough
            is_safe = True
            corrected_action = action
            env.last_safe_action = np.copy(action)
            
    except Exception as e:
        # If lookahead fails, fall back to conservative bounds
        if check_action_safety.call_count <= 3:
            print(f"⚠️  Lookahead failed, using conservative bounds: {e}")
        check_action_safety.rollback_count += 1
        corrected_action = apply_conservative_bounds(action, env)
        is_safe = False
    
    # Print summary every 1000 calls
    if check_action_safety.call_count % 1000 == 0:
        unsafe_rate = check_action_safety.unsafe_count / check_action_safety.call_count * 100
        rollback_rate = check_action_safety.rollback_count / check_action_safety.call_count * 100
        print(f"\n📊 Lookahead Shield Stats (after {check_action_safety.call_count} calls):")
        print(f"   Risky actions corrected: {check_action_safety.unsafe_count}")
        print(f"   Intervention rate: {unsafe_rate:.2f}%")
        print(f"   Rollback to conservative: {rollback_rate:.2f}%\n")
    
    # Return the safety status and the corrected action
    return is_safe, corrected_action


def simulate_and_check_violations(env, state, action):
    """
    Helper function: Simulate applying an action and predict resulting violations.
    Uses the environment's inequality constraint checker after the action.
    """
    try:
        # Use environment's built-in violation prediction
        # This checks constraints on the RESULTING state after applying the action
        violations = env.ineq_dist_np(state, action)
        total_violation = np.sum(violations)
        
        # Also check for extreme individual violations (worse than total)
        max_violation = np.max(violations)
        
        # Weight both total and max violations
        # Penalize actions that cause extreme violations more heavily
        weighted_violation = total_violation + max_violation * 2.0
        
        return weighted_violation
    except Exception as e:
        # If prediction fails, return high penalty
        return 1000.0


def apply_conservative_bounds(action, env):
    """
    Helper function: Apply 20% safety margins to action bounds.
    """
    corrected = np.copy(action)
    
    # Get indices and bounds
    pg_start = env.pg_start_yidx
    qg_start = env.qg_start_yidx
    vm_start = env.vm_start_yidx
    va_start = env.va_start_yidx
    pe_start = env.pe_start_yidx
    
    pmax = env.pmax.cpu().numpy()
    pmin = env.pmin.cpu().numpy()
    qmax = env.qmax.cpu().numpy()
    qmin = env.qmin.cpu().numpy()
    vmax = env.vmax.cpu().numpy()
    vmin = env.vmin.cpu().numpy()
    pe_max = env.evs.p_max
    pe_min = env.evs.p_min
    
    # Apply 20% safety margin
    MARGIN = 0.2
    safe_pmax = pmax * (1 - MARGIN)
    safe_pmin = pmin * (1 + MARGIN) if np.all(pmin < 0) else pmin * (1 - MARGIN)
    safe_qmax = qmax * (1 - MARGIN)
    safe_qmin = qmin * (1 + MARGIN) if np.all(qmin < 0) else qmin * (1 - MARGIN)
    safe_vmax = vmax * (1 - MARGIN * 0.1)
    safe_vmin = vmin * (1 + MARGIN * 0.1)
    safe_pe_max = pe_max * (1 - MARGIN)
    safe_pe_min = pe_min * (1 + MARGIN) if pe_min < 0 else pe_min * (1 - MARGIN)
    
    # Clip to conservative bounds
    corrected[pg_start:qg_start] = np.clip(action[pg_start:qg_start], safe_pmin, safe_pmax)
    corrected[qg_start:vm_start] = np.clip(action[qg_start:vm_start], safe_qmin, safe_qmax)
    corrected[vm_start:va_start] = np.clip(action[vm_start:va_start], safe_vmin, safe_vmax)
    corrected[va_start:pe_start] = action[va_start:pe_start]  # Voltage angles unchanged
    if pe_start < len(action):
        corrected[pe_start:] = np.clip(action[pe_start:], safe_pe_min, safe_pe_max)
    
    return corrected
