import os
import xml.etree.ElementTree as ET


def write_xml(root, path):
    """Write an ElementTree element to file with pretty-printing."""
    try:
        ET.indent(root, space="    ")
    except AttributeError:
        pass
    ET.ElementTree(root).write(path, encoding="unicode")


def build_single_intersection(net_dir):
    """
    Build SUMO network files for a single 4-way intersection.

    Topology: 4 arms (N/S/E/W), 2 lanes each, 200m long, 50 km/h.
    Traffic flows are intentionally asymmetric:
        EW axis: 500 veh/hr  (heavier)
        NS axis: 300 veh/hr  (lighter)
    This asymmetry gives the RL agent something meaningful to exploit.

    Args:
        net_dir: Directory where XML files will be written.
    """
    import subprocess

    os.makedirs(net_dir, exist_ok=True)

    # Nodes
    nodes = ET.Element("nodes")
    for nid, x, y, ntype in [
        ("center", "0",    "0",    "traffic_light"),
        ("north",  "0",    "300",  "priority"),
        ("south",  "0",    "-300", "priority"),
        ("east",   "300",  "0",    "priority"),
        ("west",   "-300", "0",    "priority"),
    ]:
        ET.SubElement(nodes, "node", id=nid, x=x, y=y, type=ntype)
    write_xml(nodes, os.path.join(net_dir, "net.nod.xml"))

    # Edges
    edges = ET.Element("edges")
    for eid, src, dst in [
        ("n2c", "north",  "center"), ("c2n", "center", "north"),
        ("s2c", "south",  "center"), ("c2s", "center", "south"),
        ("e2c", "east",   "center"), ("c2e", "center", "east"),
        ("w2c", "west",   "center"), ("c2w", "center", "west"),
    ]:
        ET.SubElement(edges, "edge", id=eid,
                      **{"from": src, "to": dst},
                      numLanes="2", speed="13.89")
    write_xml(edges, os.path.join(net_dir, "net.edg.xml"))

    # Generate net.xml
    subprocess.run([
        "netconvert",
        f"--node-files={net_dir}/net.nod.xml",
        f"--edge-files={net_dir}/net.edg.xml",
        f"--output-file={net_dir}/net.net.xml",
        "--no-warnings",
    ], check=True, capture_output=True)

    # Routes (asymmetric flows)
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", id="car", accel="2.6", decel="4.5",
                  sigma="0.5", length="5", maxSpeed="13.89")
    for rid, edges_str in [
        ("ns", "n2c c2s"), ("sn", "s2c c2n"),
        ("ew", "e2c c2w"), ("we", "w2c c2e"),
        ("ne", "n2c c2e"), ("nw", "n2c c2w"),
        ("se", "s2c c2e"), ("sw", "s2c c2w"),
        ("en", "e2c c2n"), ("es", "e2c c2s"),
        ("wn", "w2c c2n"), ("ws", "w2c c2s"),
    ]:
        ET.SubElement(routes, "route", id=rid, edges=edges_str)
    for fid, route, vph in [
        ("f_ns", "ns", "300"), ("f_sn", "sn", "300"),
        ("f_ew", "ew", "500"), ("f_we", "we", "500"),
        ("f_ne", "ne", "80"),  ("f_sw", "sw", "80"),
    ]:
        ET.SubElement(routes, "flow", id=fid, type="car", route=route,
                      begin="0", end="3600", vehsPerHour=vph)
    write_xml(routes, os.path.join(net_dir, "net.rou.xml"))

    # Config
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file",    value="net.net.xml")
    ET.SubElement(inp, "route-files", value="net.rou.xml")
    t = ET.SubElement(cfg, "time")
    ET.SubElement(t, "begin",         value="0")
    ET.SubElement(t, "end",           value="3600")
    ET.SubElement(t, "step-length",   value="1")
    rep = ET.SubElement(cfg, "report")
    ET.SubElement(rep, "no-step-log", value="true")
    ET.SubElement(rep, "no-warnings", value="true")
    write_xml(cfg, os.path.join(net_dir, "net.sumocfg"))


def build_grid_network(net_dir, scenario="uniform"):
    """
    Build SUMO network files for a 3x3 grid of intersections.

    Each intersection is a traffic light. Arms are 200m, 2 lanes, 50 km/h.

    Args:
        net_dir:   Directory where XML files will be written.
        scenario:  One of "uniform", "heavy_ew", "rush_hour".
    """
    import subprocess

    SPACING = 200
    GRID    = 3
    os.makedirs(net_dir, exist_ok=True)

    # Nodes
    nodes = ET.Element("nodes")
    for r in range(GRID):
        for c in range(GRID):
            ET.SubElement(nodes, "node",
                          id=f"I{r}{c}",
                          x=str(c * SPACING),
                          y=str(r * SPACING),
                          type="traffic_light")
    for c in range(GRID):
        ET.SubElement(nodes, "node", id=f"N{c}",
                      x=str(c * SPACING), y=str(SPACING),     type="priority")
        ET.SubElement(nodes, "node", id=f"S{c}",
                      x=str(c * SPACING), y=str(-SPACING),    type="priority")
    for r in range(GRID):
        ET.SubElement(nodes, "node", id=f"W{r}",
                      x=str(-SPACING),        y=str(r * SPACING), type="priority")
        ET.SubElement(nodes, "node", id=f"E{r}",
                      x=str(GRID * SPACING),  y=str(r * SPACING), type="priority")
    write_xml(nodes, os.path.join(net_dir, "grid.nod.xml"))

    # Edges
    edges = ET.Element("edges")

    def add_edge(eid, src, dst):
        ET.SubElement(edges, "edge", id=eid,
                      **{"from": src, "to": dst},
                      numLanes="2", speed="13.89")

    for r in range(GRID):
        for c in range(GRID - 1):
            add_edge(f"h{r}{c}r", f"I{r}{c}",   f"I{r}{c+1}")
            add_edge(f"h{r}{c}l", f"I{r}{c+1}", f"I{r}{c}")
    for r in range(GRID - 1):
        for c in range(GRID):
            add_edge(f"v{r}{c}d", f"I{r}{c}",   f"I{r+1}{c}")
            add_edge(f"v{r}{c}u", f"I{r+1}{c}", f"I{r}{c}")
    for c in range(GRID):
        add_edge(f"nIn{c}",  f"N{c}",         f"I0{c}")
        add_edge(f"nOut{c}", f"I0{c}",         f"N{c}")
        add_edge(f"sIn{c}",  f"S{c}",         f"I{GRID-1}{c}")
        add_edge(f"sOut{c}", f"I{GRID-1}{c}", f"S{c}")
    for r in range(GRID):
        add_edge(f"wIn{r}",  f"W{r}",         f"I{r}0")
        add_edge(f"wOut{r}", f"I{r}0",         f"W{r}")
        add_edge(f"eIn{r}",  f"E{r}",         f"I{r}{GRID-1}")
        add_edge(f"eOut{r}", f"I{r}{GRID-1}", f"E{r}")
    write_xml(edges, os.path.join(net_dir, "grid.edg.xml"))

    subprocess.run([
        "netconvert",
        f"--node-files={net_dir}/grid.nod.xml",
        f"--edge-files={net_dir}/grid.edg.xml",
        f"--output-file={net_dir}/grid.net.xml",
        "--no-warnings",
    ], check=True, capture_output=True)

    _build_grid_scenario(net_dir, scenario)


def _build_grid_scenario(net_dir, scenario):
    """Write route and config files for one grid traffic scenario."""
    GRID = 3

    if scenario == "uniform":
        ns_vph = sn_vph = we_vph = ew_vph = 200
    elif scenario == "heavy_ew":
        ns_vph = sn_vph = 150
        we_vph = ew_vph = 450
    elif scenario == "rush_hour":
        ns_vph = 350; sn_vph = 100
        we_vph = 300; ew_vph = 100
    else:
        ns_vph = sn_vph = we_vph = ew_vph = 200

    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", id="car", accel="2.6", decel="4.5",
                  sigma="0.5", length="5", maxSpeed="13.89")

    for c in range(GRID):
        edge_ns = " ".join([f"nIn{c}"] +
                           [f"v{r}{c}d" for r in range(GRID - 1)] +
                           [f"sOut{c}"])
        edge_sn = " ".join([f"sIn{c}"] +
                           [f"v{r}{c}u" for r in range(GRID - 2, -1, -1)] +
                           [f"nOut{c}"])
        ET.SubElement(routes, "route", id=f"ns{c}", edges=edge_ns)
        ET.SubElement(routes, "route", id=f"sn{c}", edges=edge_sn)
    for r in range(GRID):
        edge_we = " ".join([f"wIn{r}"] +
                           [f"h{r}{c}r" for c in range(GRID - 1)] +
                           [f"eOut{r}"])
        edge_ew = " ".join([f"eIn{r}"] +
                           [f"h{r}{c}l" for c in range(GRID - 2, -1, -1)] +
                           [f"wOut{r}"])
        ET.SubElement(routes, "route", id=f"we{r}", edges=edge_we)
        ET.SubElement(routes, "route", id=f"ew{r}", edges=edge_ew)

    fid = 0
    for c in range(GRID):
        ET.SubElement(routes, "flow", id=f"f{fid}", type="car",
                      route=f"ns{c}", begin="0", end="3600",
                      vehsPerHour=str(ns_vph)); fid += 1
        ET.SubElement(routes, "flow", id=f"f{fid}", type="car",
                      route=f"sn{c}", begin="0", end="3600",
                      vehsPerHour=str(sn_vph)); fid += 1
    for r in range(GRID):
        ET.SubElement(routes, "flow", id=f"f{fid}", type="car",
                      route=f"we{r}", begin="0", end="3600",
                      vehsPerHour=str(we_vph)); fid += 1
        ET.SubElement(routes, "flow", id=f"f{fid}", type="car",
                      route=f"ew{r}", begin="0", end="3600",
                      vehsPerHour=str(ew_vph)); fid += 1

    write_xml(routes, os.path.join(net_dir, f"grid_{scenario}.rou.xml"))

    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file",    value="grid.net.xml")
    ET.SubElement(inp, "route-files", value=f"grid_{scenario}.rou.xml")
    t = ET.SubElement(cfg, "time")
    ET.SubElement(t, "begin",         value="0")
    ET.SubElement(t, "end",           value="3600")
    ET.SubElement(t, "step-length",   value="1")
    rep = ET.SubElement(cfg, "report")
    ET.SubElement(rep, "no-step-log", value="true")
    ET.SubElement(rep, "no-warnings", value="true")
    write_xml(cfg, os.path.join(net_dir, f"grid_{scenario}.sumocfg"))
