# shield_model.py - Conservative Bounds Safety Shield (Option 2C)

import numpy as np

def check_action_safety(state, action, env):
    """
    Predictive safety shield that checks if an action will cause constraint violations.
    Uses the environment's constraint checking functions to predict violations.
    """
    
    # 1. Create a copy of the action to modify
    corrected_action = np.copy(action)
    
    # Debug: Track if this is being called
    if not hasattr(check_action_safety, 'call_count'):
        check_action_safety.call_count = 0
        check_action_safety.unsafe_count = 0
        print("🛡️ Conservative Bounds Shield initialized (MARGIN=0.10)")
    check_action_safety.call_count += 1
    
    if check_action_safety.call_count == 1:
        print(f"🛡️ Shield called for first time! Action shape: {action.shape}")
    
    # 2. CONSERVATIVE BOUNDS SHIELD: Apply safety margins to action bounds
    # This creates a "safe zone" well within the actual physical limits
    
    # Get the safety limits from environment
    ng = env.ng
    nbus = env.nbus
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
    
    # CONSERVATIVE BOUNDS: Apply 10% safety margin
    SAFETY_MARGIN = 0.10
    safe_pmax = pmax * (1 - SAFETY_MARGIN)
    safe_pmin = pmin * (1 + SAFETY_MARGIN) if np.all(pmin < 0) else pmin * (1 - SAFETY_MARGIN)
    safe_qmax = qmax * (1 - SAFETY_MARGIN)
    safe_qmin = qmin * (1 + SAFETY_MARGIN) if np.all(qmin < 0) else qmin * (1 - SAFETY_MARGIN)
    safe_vmax = vmax * (1 - SAFETY_MARGIN * 0.5)  # Smaller margin for voltage (5%)
    safe_vmin = vmin * (1 + SAFETY_MARGIN * 0.5)
    safe_pe_max = pe_max * (1 - SAFETY_MARGIN)
    safe_pe_min = pe_min * (1 + SAFETY_MARGIN) if pe_min < 0 else pe_min * (1 - SAFETY_MARGIN)
    
    # Clip actions to conservative bounds
    corrected_action[pg_start:qg_start] = np.clip(action[pg_start:qg_start], safe_pmin, safe_pmax)
    corrected_action[qg_start:vm_start] = np.clip(action[qg_start:vm_start], safe_qmin, safe_qmax)
    corrected_action[vm_start:va_start] = np.clip(action[vm_start:va_start], safe_vmin, safe_vmax)
    
    # For voltage angles and EV charging, use standard clipping (no special bounds)
    corrected_action[va_start:pe_start] = action[va_start:pe_start]  # Copy voltage angles as-is
    if pe_start < len(action):
        corrected_action[pe_start:] = np.clip(action[pe_start:], safe_pe_min, safe_pe_max)
    
    # 5. Check if any corrections were made
    if not np.allclose(action, corrected_action, rtol=1e-5):
        check_action_safety.unsafe_count += 1
        is_safe = False
        
        if check_action_safety.unsafe_count <= 10:
            max_diff = np.max(np.abs(action - corrected_action))
            print(f"🛡️ CONSERVATIVE BOUNDS: Corrected action (#{check_action_safety.unsafe_count})")
            print(f"   Max action change: {max_diff:.4f}")
            print(f"   Applied {SAFETY_MARGIN*100:.0f}% safety margin")
    else:
        is_safe = True
    
    # Print summary every 1000 calls
    if check_action_safety.call_count % 1000 == 0:
        unsafe_rate = check_action_safety.unsafe_count / check_action_safety.call_count * 100
        print(f"\n📊 Shield Stats (after {check_action_safety.call_count} calls):")
        print(f"   Unsafe actions corrected: {check_action_safety.unsafe_count}")
        print(f"   Intervention rate: {unsafe_rate:.2f}%\n")
    
    # 6. Return the safety status and the corrected action
    return is_safe, corrected_action