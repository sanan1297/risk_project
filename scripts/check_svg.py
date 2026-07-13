import re

svg = open("docs/diagrams/8_7_pipeline_completo_snake.svg").read()

# In Mermaid SVG, nodes are rects with class="label-container"
node_blocks = re.findall(r'<g class="label-container">(.*?)</g>', svg, re.DOTALL)

print("Found label containers:", len(node_blocks))

# Try a different approach - find all <g> with transform in the main svg
transforms = re.findall(r'<g transform="translate\(([\d.]+),([\d.]+)\)"', svg)
print(f"Found {len(transforms)} transforms")

# Find foreignObject elements (Mermaid uses these for labels)
import xml.etree.ElementTree as ET
root = ET.fromstring(svg)
ns = {'svg': 'http://www.w3.org/2000/svg'}

# Find all nodes by looking for rect + text combinations
for i, g in enumerate(root.findall('.//svg:g', ns)):
    title_el = g.find('svg:title', ns)
    title = title_el.text if title_el is not None else ""
    transform = g.get('transform', '')
    if title and 'flowchart' not in title and title.strip():
        print(f"g[{i}]: title='{title}' transform='{transform}'")
