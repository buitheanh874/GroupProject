from __future__ import annotations


def test_rho_min_enforcement():
    action_splits = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]
    
    C = 60
    rho_min = 0.1
    g_min = int(rho_min * C)
    
    for i, (rho_ns, rho_ew) in enumerate(action_splits):
        g_ns = int(round(rho_ns * C))
        g_ns = max(g_min, g_ns)
        g_ns = min(g_ns, max(g_min, C - g_min))
        g_ew = C - g_ns
        
        assert g_ns >= g_min, f"Action {i}: g_ns={g_ns} < {g_min}"
        assert g_ew >= g_min, f"Action {i}: g_ew={g_ew} < {g_min}"
        assert g_ns + g_ew == C, f"Action {i}: g_ns+g_ew != {C}"
        
        print(f"Action {i}: ({rho_ns:.1f}, {rho_ew:.1f}) -> g_ns={g_ns}s, g_ew={g_ew}s")


if __name__ == "__main__":
    test_rho_min_enforcement()
    print("\nAll action space tests passed")