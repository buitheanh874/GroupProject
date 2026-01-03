Hub-and-spoke network placeholder.

Provide the following SUMO files before running multi-TLS configs:
- hub_spoke.net.xml
- hub_spoke.rou.xml
- hub_spoke.sumocfg

Each TLS should be named Center, N, E, S, W. Include lane mappings per TLS in the config (lane_groups_by_tls) and downstream links for Center (N/E/S/W keys).
