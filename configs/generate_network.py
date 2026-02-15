"""
Generate a 4-arm signalized intersection network for DNLight.

Creates:
  - single_intersection.net.xml  (network with TLS)
  - single_intersection.rou.xml  (routes + vehicle demand with EMVs)
  - single_intersection.sumocfg  (simulation config)

Run: python configs/generate_network.py
"""
import os
import sys
import random
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Network parameters ──────────────────────────────────────────────
ARM_LENGTH = 200        # metres per arm (detection range)
LANES_PER_DIR = 2       # lanes per direction
SPEED_LIMIT = 13.89     # m/s  (~50 km/h)
NET_FILE = os.path.join(SCRIPT_DIR, "single_intersection.net.xml")

# ── Route / demand parameters ───────────────────────────────────────
ROU_FILE = os.path.join(SCRIPT_DIR, "single_intersection.rou.xml")
SIM_END = 3600          # seconds
SOCIAL_VPH = 600        # social vehicles per hour (total, split across routes)
NUM_EMVS = 8            # total emergency vehicles
SEED = 42

# ── TLS phases (yellow = 3 s each, green = variable) ────────────────
# Phase order:  NS‑straight → yellow → NS‑left → yellow →
#               EW‑straight → yellow → EW‑left  → yellow
YELLOW_DUR = 3
GREEN_DUR = 30  # default green per phase


def generate_node_file(path):
    """Write .nod.xml with 5 nodes: centre + 4 arms."""
    root = ET.Element("nodes")
    # Centre (traffic‑light controlled)
    ET.SubElement(root, "node", id="C", x="0", y="0", type="traffic_light")
    # Arms
    ET.SubElement(root, "node", id="N", x="0",              y=str(ARM_LENGTH),  type="priority")
    ET.SubElement(root, "node", id="S", x="0",              y=str(-ARM_LENGTH), type="priority")
    ET.SubElement(root, "node", id="E", x=str(ARM_LENGTH),  y="0",              type="priority")
    ET.SubElement(root, "node", id="W", x=str(-ARM_LENGTH), y="0",              type="priority")
    _write_xml(root, path)


def generate_edge_file(path):
    """Write .edg.xml with 8 edges (2 per arm, in + out)."""
    root = ET.Element("edges")
    for d in ["N", "S", "E", "W"]:
        # Incoming edge toward centre
        ET.SubElement(root, "edge", id=f"{d}_to_C", **{
            "from": d, "to": "C",
            "numLanes": str(LANES_PER_DIR),
            "speed": str(SPEED_LIMIT)
        })
        # Outgoing edge away from centre
        ET.SubElement(root, "edge", id=f"C_to_{d}", **{
            "from": "C", "to": d,
            "numLanes": str(LANES_PER_DIR),
            "speed": str(SPEED_LIMIT)
        })
    _write_xml(root, path)


def generate_tls_file(path):
    """
    Write .tll.xml – traffic‑light logic for node C.

    4 green phases + 4 yellow transitions = 8 phases total.
    Lane indexing (SUMO default for 2‑lane approach, 4 arms):
        N_to_C: lanes 0,1   S_to_C: lanes 0,1
        E_to_C: lanes 0,1   W_to_C: lanes 0,1

    State string order follows SUMO's internal connection ordering.
    We'll let netconvert compute the exact state strings from the
    high-level phase definitions. Here we use a simplified approach
    with explicit connection-based states.
    """
    root = ET.Element("tlLogics")
    tls = ET.SubElement(root, "tlLogic", id="C", type="static",
                        programID="dnlight", offset="0")
    # We define 8 phases. SUMO has one char per connection.
    # For a 4-arm, 2-lane intersection the number of connections varies.
    # We'll use a placeholder here and let netconvert auto-generate
    # the correct phase states. Instead, we pass phase info via
    # additional-file or let netconvert figure it out.
    #
    # Simpler approach: generate with netconvert and then
    # patch the TLS program via traci at runtime.
    #
    # For now, just generate the network and we will control phases
    # via traci.setPhase() in the environment.
    _write_xml(root, path)


def build_network():
    """Run netconvert to produce .net.xml."""
    nod = os.path.join(SCRIPT_DIR, "_tmp.nod.xml")
    edg = os.path.join(SCRIPT_DIR, "_tmp.edg.xml")
    tll = os.path.join(SCRIPT_DIR, "_tmp.tll.xml")

    generate_node_file(nod)
    generate_edge_file(edg)
    generate_tls_file(tll)

    cmd = [
        "netconvert",
        "--node-files", nod,
        "--edge-files", edg,
        # "--tllogic-files", tll,   # let SUMO auto-generate TLS
        "--output-file", NET_FILE,
        "--no-turnarounds", "true",
        "--junctions.join", "false",
        "--tls.guess", "true",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Cleanup temp files
    for f in [nod, edg, tll]:
        os.remove(f)
    print(f"Network written to {NET_FILE}")


def generate_routes():
    """
    Generate .rou.xml with social vehicles and EMVs.
    Routes go from each arm to every other arm (12 OD pairs).
    """
    random.seed(SEED)

    root = ET.Element("routes")

    # ── Define routes (12 origin‑destination pairs) ──────────────
    directions = ["N", "S", "E", "W"]
    routes = []
    for orig in directions:
        for dest in directions:
            if orig == dest:
                continue
            rid = f"route_{orig}_{dest}"
            edges = f"{orig}_to_C C_to_{dest}"
            ET.SubElement(root, "route", id=rid, edges=edges)
            routes.append(rid)

    # ── Social vehicles ──────────────────────────────────────────
    interval = SIM_END / (SOCIAL_VPH)
    vid = 0
    for t_idx in range(SOCIAL_VPH):
        depart = round(t_idx * interval + random.uniform(-1, 1), 1)
        depart = max(0, min(depart, SIM_END - 1))
        route = random.choice(routes)
        ET.SubElement(root, "vehicle", id=f"car_{vid}", type="car",
                      route=route, depart=str(depart), departLane="best",
                      departSpeed="max")
        vid += 1

    # ── Emergency vehicles ───────────────────────────────────────
    emv_types = ["ambulance", "fire_truck", "police"]
    for i in range(NUM_EMVS):
        depart = round(random.uniform(60, SIM_END - 300), 1)
        route = random.choice(routes)
        etype = emv_types[i % len(emv_types)]
        ET.SubElement(root, "vehicle", id=f"emv_{i}", type=etype,
                      route=route, depart=str(depart), departLane="best",
                      departSpeed="max")

    # Sort vehicles by departure time
    vehicles = [e for e in root if e.tag == "vehicle"]
    route_elems = [e for e in root if e.tag == "route"]
    root.clear()
    for r in route_elems:
        root.append(r)
    for v in sorted(vehicles, key=lambda v: float(v.get("depart"))):
        root.append(v)

    _write_xml(root, ROU_FILE)
    print(f"Routes written to {ROU_FILE} "
          f"({SOCIAL_VPH} cars + {NUM_EMVS} EMVs)")


def generate_sumocfg():
    """Write .sumocfg binding network, routes, and additional files."""
    cfg_path = os.path.join(SCRIPT_DIR, "single_intersection.sumocfg")
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value="single_intersection.net.xml")
    ET.SubElement(inp, "route-files", value="single_intersection.rou.xml")
    ET.SubElement(inp, "additional-files", value="vehicle_types.add.xml")

    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value="0")
    ET.SubElement(time, "end", value=str(SIM_END))

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", value="-1")

    _write_xml(root, cfg_path)
    print(f"Config written to {cfg_path}")


def _write_xml(root, path):
    """Pretty‑print an ElementTree to file."""
    rough = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dom.toprettyxml(indent="    "))


if __name__ == "__main__":
    print("=== DNLight Network Generator ===")
    build_network()
    generate_routes()
    generate_sumocfg()
    print("Done! All files in:", SCRIPT_DIR)
