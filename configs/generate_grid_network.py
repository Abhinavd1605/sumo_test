"""
Generate a 2x2 grid of signalized intersections for DNLight multi-agent.

Creates:
  - grid_network.net.xml   (4 intersections in a grid)
  - grid_network.rou.xml   (routes with social vehicles + EMVs)
  - grid_network.sumocfg   (simulation config)

Run: python configs/generate_grid_network.py
"""
import os
import sys
import random
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Network parameters ──────────────────────────────────────────────
GRID_ROWS = 2
GRID_COLS = 2
BLOCK_LENGTH = 200       # metres between intersections
ARM_LENGTH = 200         # metres for outer arms
LANES_PER_DIR = 2
SPEED_LIMIT = 13.89      # m/s (~50 km/h)
NET_FILE = os.path.join(SCRIPT_DIR, "grid_network.net.xml")

# ── Route / demand parameters ───────────────────────────────────────
ROU_FILE = os.path.join(SCRIPT_DIR, "grid_network.rou.xml")
SIM_END = 3600           # seconds
SOCIAL_VPH = 1200        # social vehicles per hour (total)
NUM_EMVS = 20            # emergency vehicles
SEED = 42


def generate_node_file(path):
    """Write .nod.xml with grid intersection nodes + boundary nodes."""
    root = ET.Element("nodes")

    # Grid intersection nodes (traffic-light controlled)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nid = f"C{r}{c}"
            x = c * BLOCK_LENGTH
            y = r * BLOCK_LENGTH
            ET.SubElement(root, "node", id=nid,
                          x=str(x), y=str(y), type="traffic_light")

    # Boundary nodes (outer ends of the grid)
    for c in range(GRID_COLS):
        x = c * BLOCK_LENGTH
        # South boundary
        ET.SubElement(root, "node", id=f"S{c}",
                      x=str(x), y=str(-ARM_LENGTH), type="priority")
        # North boundary
        ET.SubElement(root, "node", id=f"N{c}",
                      x=str(x), y=str((GRID_ROWS - 1) * BLOCK_LENGTH + ARM_LENGTH),
                      type="priority")

    for r in range(GRID_ROWS):
        y = r * BLOCK_LENGTH
        # West boundary
        ET.SubElement(root, "node", id=f"W{r}",
                      x=str(-ARM_LENGTH), y=str(y), type="priority")
        # East boundary
        ET.SubElement(root, "node", id=f"E{r}",
                      x=str((GRID_COLS - 1) * BLOCK_LENGTH + ARM_LENGTH),
                      y=str(y), type="priority")

    _write_xml(root, path)


def generate_edge_file(path):
    """Write .edg.xml with internal grid edges + boundary edges."""
    root = ET.Element("edges")

    edge_attrs = {"numLanes": str(LANES_PER_DIR), "speed": str(SPEED_LIMIT)}

    # Internal grid edges (horizontal and vertical)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nid = f"C{r}{c}"
            # East neighbor
            if c + 1 < GRID_COLS:
                eid_right = f"C{r}{c + 1}"
                ET.SubElement(root, "edge", id=f"{nid}_to_{eid_right}",
                              **{"from": nid, "to": eid_right}, **edge_attrs)
                ET.SubElement(root, "edge", id=f"{eid_right}_to_{nid}",
                              **{"from": eid_right, "to": nid}, **edge_attrs)
            # North neighbor
            if r + 1 < GRID_ROWS:
                nid_up = f"C{r + 1}{c}"
                ET.SubElement(root, "edge", id=f"{nid}_to_{nid_up}",
                              **{"from": nid, "to": nid_up}, **edge_attrs)
                ET.SubElement(root, "edge", id=f"{nid_up}_to_{nid}",
                              **{"from": nid_up, "to": nid}, **edge_attrs)

    # Boundary edges
    for c in range(GRID_COLS):
        # South boundary to bottom row
        bot = f"C0{c}"
        ET.SubElement(root, "edge", id=f"S{c}_to_{bot}",
                      **{"from": f"S{c}", "to": bot}, **edge_attrs)
        ET.SubElement(root, "edge", id=f"{bot}_to_S{c}",
                      **{"from": bot, "to": f"S{c}"}, **edge_attrs)
        # North boundary to top row
        top = f"C{GRID_ROWS - 1}{c}"
        ET.SubElement(root, "edge", id=f"N{c}_to_{top}",
                      **{"from": f"N{c}", "to": top}, **edge_attrs)
        ET.SubElement(root, "edge", id=f"{top}_to_N{c}",
                      **{"from": top, "to": f"N{c}"}, **edge_attrs)

    for r in range(GRID_ROWS):
        # West boundary to left column
        left = f"C{r}0"
        ET.SubElement(root, "edge", id=f"W{r}_to_{left}",
                      **{"from": f"W{r}", "to": left}, **edge_attrs)
        ET.SubElement(root, "edge", id=f"{left}_to_W{r}",
                      **{"from": left, "to": f"W{r}"}, **edge_attrs)
        # East boundary to right column
        right = f"C{r}{GRID_COLS - 1}"
        ET.SubElement(root, "edge", id=f"E{r}_to_{right}",
                      **{"from": f"E{r}", "to": right}, **edge_attrs)
        ET.SubElement(root, "edge", id=f"{right}_to_E{r}",
                      **{"from": right, "to": f"E{r}"}, **edge_attrs)

    _write_xml(root, path)


def build_network():
    """Run netconvert to produce .net.xml."""
    nod = os.path.join(SCRIPT_DIR, "_tmp_grid.nod.xml")
    edg = os.path.join(SCRIPT_DIR, "_tmp_grid.edg.xml")

    generate_node_file(nod)
    generate_edge_file(edg)

    cmd = [
        "netconvert",
        "--node-files", nod,
        "--edge-files", edg,
        "--output-file", NET_FILE,
        "--no-turnarounds", "true",
        "--junctions.join", "false",
        "--tls.guess", "true",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Cleanup
    for f in [nod, edg]:
        os.remove(f)
    print(f"Grid network written to {NET_FILE}")


def generate_routes():
    """Generate valid routes across the 2x2 grid using BFS pathfinding."""
    random.seed(SEED)
    
    # 1. Build adjacency list of edges
    adj = {}
    edges_list = []
    
    # Internal grid edges
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nid = f"C{r}{c}"
            if c + 1 < GRID_COLS:
                nxt = f"C{r}{c+1}"
                edges_list.append((nid, nxt, f"{nid}_to_{nxt}"))
                edges_list.append((nxt, nid, f"{nxt}_to_{nid}"))
            if r + 1 < GRID_ROWS:
                nxt = f"C{r+1}{c}"
                edges_list.append((nid, nxt, f"{nid}_to_{nxt}"))
                edges_list.append((nxt, nid, f"{nxt}_to_{nid}"))
                
    # Boundary edges
    for c in range(GRID_COLS):
        bot = f"C0{c}"
        edges_list.append((f"S{c}", bot, f"S{c}_to_{bot}"))
        edges_list.append((bot, f"S{c}", f"{bot}_to_S{c}"))
        top = f"C{GRID_ROWS-1}{c}"
        edges_list.append((f"N{c}", top, f"N{c}_to_{top}"))
        edges_list.append((top, f"N{c}", f"{top}_to_N{c}"))
    for r in range(GRID_ROWS):
        left = f"C{r}0"
        edges_list.append((f"W{r}", left, f"W{r}_to_{left}"))
        edges_list.append((left, f"W{r}", f"{left}_to_W{r}"))
        right = f"C{r}{GRID_COLS-1}"
        edges_list.append((f"E{r}", right, f"E{r}_to_{right}"))
        edges_list.append((right, f"E{r}", f"{right}_to_E{r}"))
        
    for u, v, eid in edges_list:
        if u not in adj: adj[u] = []
        adj[u].append((v, eid))

    def find_path(start_node, end_node):
        """BFS to find shortest path of edges."""
        queue = [(start_node, [])]
        visited = {start_node}
        while queue:
            node, path = queue.pop(0)
            if node == end_node:
                return path
            for neighbor, eid in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [eid]))
        return None

    # 2. Define all valid orig-dest pairs (boundary to boundary)
    boundary_nodes = (
        [f"S{c}" for c in range(GRID_COLS)] +
        [f"N{c}" for c in range(GRID_COLS)] +
        [f"W{r}" for r in range(GRID_ROWS)] +
        [f"E{r}" for r in range(GRID_ROWS)]
    )
    
    routes_data = []
    for orig in boundary_nodes:
        for dest in boundary_nodes:
            if orig == dest: continue
            path = find_path(orig, dest)
            if path:
                routes_data.append(" ".join(path))

    if not routes_data:
        print("ERROR: No routes found! Check network topology.")
        return

    # 3. Create route elements
    root = ET.Element("routes")
    route_ids = []
    for idx, r_edges in enumerate(routes_data):
        rid = f"route_{idx}"
        ET.SubElement(root, "route", id=rid, edges=r_edges)
        route_ids.append(rid)

    # 4. Generate vehicle demand
    interval = SIM_END / SOCIAL_VPH
    vehicles = []
    for i in range(SOCIAL_VPH):
        depart = round(i * interval + random.uniform(-1, 1), 1)
        depart = max(0, min(depart, SIM_END - 1))
        vehicles.append({
            'id': f"car_{i}", 'type': "car", 'route': random.choice(route_ids),
            'depart': depart, 'departLane': "best", 'departSpeed': "max"
        })

    emv_types = ["ambulance", "fire_truck", "police"]
    for i in range(NUM_EMVS):
        depart = round(random.uniform(60, SIM_END - 300), 1)
        vehicles.append({
            'id': f"emv_{i}", 'type': random.choice(emv_types),
            'route': random.choice(route_ids), 'depart': depart,
            'departLane': "best", 'departSpeed': "max"
        })

    # Sort and add to XML
    for v in sorted(vehicles, key=lambda x: x['depart']):
        ET.SubElement(root, "vehicle", id=v['id'], type=v['type'],
                      route=v['route'], depart=str(v['depart']),
                      departLane=v['departLane'], departSpeed=v['departSpeed'])

    _write_xml(root, ROU_FILE)
    print(f"Routes written to {ROU_FILE} ({len(vehicles)} vehicles)")



def generate_sumocfg():
    """Write .sumocfg for the grid network."""
    cfg_path = os.path.join(SCRIPT_DIR, "grid_network.sumocfg")
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value="grid_network.net.xml")
    ET.SubElement(inp, "route-files", value="grid_network.rou.xml")
    ET.SubElement(inp, "additional-files", value="vehicle_types.add.xml")

    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value=str(SIM_END))

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "time-to-teleport", value="-1")

    _write_xml(root, cfg_path)
    print(f"Config written to {cfg_path}")


def _write_xml(root, path):
    """Pretty-print an ElementTree to file."""
    rough = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dom.toprettyxml(indent="    "))


if __name__ == "__main__":
    print("=== DNLight Grid Network Generator ===")
    build_network()
    generate_routes()
    generate_sumocfg()
    print("Done! All files in:", SCRIPT_DIR)
