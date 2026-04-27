# simulation_shield.py - PyPower-based Predictive Safety Shield
"""
This shield uses actual PyPower power flow simulation to predict
whether an action will cause constraint violations BEFORE applying it.

This is more accurate than heuristic-based shields because it uses
the same physics solver (pp.runpf) that the environment uses.
"""

import numpy as np
import copy
try:
    import pypower.api as pp
    PYPOWER_AVAILABLE = True
except ImportError:
    PYPOWER_AVAILABLE = False
    print("⚠️  PyPower not available, simulation shield will use fallback mode")


def check_action_safety(state, action, env):
    """
    Simulation-based safety shield using PyPower power flow solver.
    
    Returns:
        is_safe (bool): Whether the action is safe
        corrected_action (np.ndarray): Safe action to use
    """
    # Initialize tracking
    if not hasattr(check_action_safety, 'call_count'):
        check_action_safety.call_count = 0
        check_action_safety.unsafe_count = 0
        check_action_safety.simulation_failures = 0
        print("🛡️ Simulation-Based Shield initialized (using PyPower runpf)")
    check_action_safety.call_count += 1
    
    if check_action_safety.call_count == 1:
        print(f"🛡️ Shield called for first time! Action shape: {action.shape}")
        print(f"🛡️ PyPower available: {PYPOWER_AVAILABLE}")
    
    # If PyPower not available, fall back to conservative bounds
    if not PYPOWER_AVAILABLE:
        return apply_conservative_bounds_fallback(action, env)
    
    # Try progressively more conservative actions until we find a safe one
    candidates = [
        ("original", action, 1.0),
        ("scaled_90%", action * 0.9, 0.9),
        ("scaled_80%", action * 0.8, 0.8),
        ("scaled_70%", action * 0.7, 0.7),
        ("scaled_50%", action * 0.5, 0.5),
        ("conservative", apply_conservative_bounds_fallback(action, env)[1], 0.0),
    ]
    
    for name, candidate_action, scale in candidates:
        is_safe = simulate_power_flow_safety(candidate_action, env)
        
        if is_safe:
            if name != "original":
                check_action_safety.unsafe_count += 1
                if check_action_safety.unsafe_count <= 10:
                    print(f"🛡️ SIMULATION SHIELD: Blocked unsafe action (#{check_action_safety.unsafe_count})")
                    print(f"   Safe alternative found: {name}")
            
            # Print summary every 1000 calls
            if check_action_safety.call_count % 1000 == 0:
                unsafe_rate = check_action_safety.unsafe_count / check_action_safety.call_count * 100
                fail_rate = check_action_safety.simulation_failures / check_action_safety.call_count * 100
                print(f"\n📊 Simulation Shield Stats (after {check_action_safety.call_count} calls):")
                print(f"   Unsafe actions corrected: {check_action_safety.unsafe_count}")
                print(f"   Intervention rate: {unsafe_rate:.2f}%")
                print(f"   Simulation failures: {fail_rate:.2f}%\n")
            
            return (name == "original"), candidate_action
    
    # If no candidate is safe, use conservative bounds as last resort
    check_action_safety.unsafe_count += 1
    check_action_safety.simulation_failures += 1
    return False, apply_conservative_bounds_fallback(action, env)[1]


def simulate_power_flow_safety(action, env):
    """
    Simulate applying an action using PyPower and check for violations.
    
    Returns:
        bool: True if action is safe (no violations), False otherwise
    """
    try:
        # Get the base environment (unwrap if needed)
        raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Create a virtual copy of the power system
        virtual_ppc = copy.deepcopy(raw_env.ppc)
        
        # Extract action components
        ng = raw_env.ng
        nbus = raw_env.nbus
        pg_start = raw_env.pg_start_yidx
        qg_start = raw_env.qg_start_yidx
        vm_start = raw_env.vm_start_yidx
        va_start = raw_env.va_start_yidx
        
        pg_action = action[pg_start:qg_start]
        qg_action = action[qg_start:vm_start]
        vm_action = action[vm_start:va_start]
        va_action = action[va_start:] if va_start < len(action) else None
        
        # Apply action to virtual PPC
        # Note: Actions are in per-unit, PPC uses MW/MVA
        baseMVA = virtual_ppc['baseMVA']
        
        # Update generator setpoints (columns: Pg=1, Qg=2 in PyPower 0-indexed)
        virtual_ppc['gen'][:ng, 1] = pg_action * baseMVA  # Pg in MW
        virtual_ppc['gen'][:ng, 2] = qg_action * baseMVA  # Qg in MVAr
        
        # Update voltage setpoints (column: Vm=5)
        # Map voltage actions to generator buses
        gen_buses = virtual_ppc['gen'][:ng, 0].astype(int)  # Bus numbers
        for i, bus_num in enumerate(gen_buses):
            bus_idx = int(bus_num) - 1  # Convert to 0-indexed
            if bus_idx < len(vm_action):
                virtual_ppc['bus'][bus_idx, 7] = vm_action[bus_idx]  # Vm setpoint
        
        # Run power flow with fast settings
        ppopt = pp.ppoption(
            PF_ALG=1,  # Newton's method
            MAX_IT=10,  # Fast convergence attempt
            ENFORCE_Q_LIMS=1,  # Enforce Q limits
            VERBOSE=0,  # Silent
            OUT_ALL=0   # No output
        )
        
        results, success = pp.runpf(virtual_ppc, ppopt)
        
        if not success:
            # Power flow didn't converge = UNSAFE (system instability)
            return False
        
        # Check resulting voltages for violations
        vm_result = results['bus'][:, 7]  # Voltage magnitudes
        vmax = results['bus'][:, 11]  # Upper limits
        vmin = results['bus'][:, 12]  # Lower limits
        
        # Allow small tolerance (0.5%)
        TOLERANCE = 0.005
        voltage_violations = np.any(vm_result > vmax * (1 + TOLERANCE)) or \
                           np.any(vm_result < vmin * (1 - TOLERANCE))
        
        if voltage_violations:
            return False
        
        # Check generator limits
        pg_result = results['gen'][:, 1] / baseMVA  # Convert back to p.u.
        qg_result = results['gen'][:, 2] / baseMVA
        
        pmax = results['gen'][:, 8] / baseMVA
        pmin = results['gen'][:, 9] / baseMVA
        qmax = results['gen'][:, 3] / baseMVA
        qmin = results['gen'][:, 4] / baseMVA
        
        gen_violations = np.any(pg_result > pmax * (1 + TOLERANCE)) or \
                        np.any(pg_result < pmin * (1 - TOLERANCE)) or \
                        np.any(qg_result > qmax * (1 + TOLERANCE)) or \
                        np.any(qg_result < qmin * (1 - TOLERANCE))
        
        if gen_violations:
            return False
        
        # Action is SAFE
        return True
        
    except Exception as e:
        # If simulation fails for any reason, assume unsafe
        if check_action_safety.call_count <= 3:
            print(f"⚠️  Simulation failed: {e}")
        return False


def apply_conservative_bounds_fallback(action, env):
    """
    Fallback shield: Apply 20% safety margins when simulation unavailable.
    """
    corrected = np.copy(action)
    
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
    
    MARGIN = 0.2
    safe_pmax = pmax * (1 - MARGIN)
    safe_pmin = pmin * (1 + MARGIN) if np.all(pmin < 0) else pmin * (1 - MARGIN)
    safe_qmax = qmax * (1 - MARGIN)
    safe_qmin = qmin * (1 + MARGIN) if np.all(qmin < 0) else qmin * (1 - MARGIN)
    safe_vmax = vmax * (1 - MARGIN * 0.1)
    safe_vmin = vmin * (1 + MARGIN * 0.1)
    
    corrected[pg_start:qg_start] = np.clip(action[pg_start:qg_start], safe_pmin, safe_pmax)
    corrected[qg_start:vm_start] = np.clip(action[qg_start:vm_start], safe_qmin, safe_qmax)
    corrected[vm_start:va_start] = np.clip(action[vm_start:va_start], safe_vmin, safe_vmax)
    
    is_safe = not np.allclose(action, corrected, rtol=1e-5)
    return is_safe, corrected
