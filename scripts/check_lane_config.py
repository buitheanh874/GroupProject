import traci

traci.start(['sumo', '-n', 'networks/BIGNET.net.xml', '--no-step-log'])

network_lanes = set(traci.lane.getIDList())

config_lanes = {
    'J0': {
        'ns': ['-E3_0', '-E3_1', '-E2_0', '-E2_1'],
        'ew': ['-E1_0', '-E1_1', '-E1_2', '-E0_0', '-E0_1', '-E0_2']
    },
    'J3': {
        'ns': ['-E25_0', '-E25_1', 'E2_0', 'E2_1'],
        'ew': ['-E26_0', 'E24_0']
    },
    'J1': {
        'ns': ['-E6_0', '-E6_1', '-E5_0', '-E5_1'],
        'ew': ['E0_0', 'E0_1', 'E0_2', '-E4_0', '-E4_1', '-E4_2']
    },
    'J4': {
        'ns': ['E3_0', 'E3_1', '-E16_0', '-E16_1'],
        'ew': ['E14_0', '-E15_0']
    },
    'J2': {
        'ns': ['-E19_0', '-E19_1', '-E18_0', '-E18_1'],
        'ew': ['E1_0', 'E1_1', 'E1_2', '-E17_0', '-E17_1', '-E17_2']
    },
    'J7': {
        'ns': ['-E27_0', '-E27_1', 'E6_0', 'E6_1'],
        'ew': ['-E28_0', 'E26_0']
    },
    'J6': {
        'ns': ['E5_0', 'E5_1', '-E13_0', '-E13_1'],
        'ew': ['-E12_0', '-E14_0']
    },
    'J14': {
        'ns': ['E18_0', 'E18_1', '-E21_0', '-E21_1'],
        'ew': ['E15_0', '-E20_0']
    },
    'J17': {
        'ns': ['-E22_0', '-E22_1', 'E19_0', 'E19_1'],
        'ew': ['-E24_0', '-E23_0']
    }
}

print("=" * 60)
print("LANE CONFIGURATION VALIDATION")
print("=" * 60)

missing_lanes = []
for tls_id, directions in config_lanes.items():
    print(f"\n[{tls_id}]")
    for direction, lanes in directions.items():
        for lane in lanes:
            exists = lane in network_lanes
            status = "OK" if exists else "MISSING"
            print(f"  {direction}: {lane} -> {status}")
            if not exists:
                missing_lanes.append((tls_id, direction, lane))

print("\n" + "=" * 60)
if missing_lanes:
    print(f"FOUND {len(missing_lanes)} MISSING LANES:")
    for tls, dir, lane in missing_lanes:
        print(f"  {tls}/{dir}: {lane}")
else:
    print("ALL LANES EXIST IN NETWORK")
print("=" * 60)

traci.close()
