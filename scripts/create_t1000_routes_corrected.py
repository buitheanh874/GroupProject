#!/usr/bin/env python3
"""
Create corrected T1000 route files that maintain original perHour but change end from 3600 to 1000.
This gives us 1000s injection period with CORRECT demand levels (not 3.6x scaled).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

def create_t1000_routes(demand):
    """Create T1000 route files for a given demand level."""
    base_dir = Path(f"networks/variants/train_turn801010/{demand}")
    
    # Find all original route files
    original_files = list(base_dir.glob(f"*_d{demand}.rou.xml"))
    original_files = [f for f in original_files if "_t1000" not in f.name]
    
    print(f"\nProcessing demand {demand}: found {len(original_files)} files")
    
    for orig_file in original_files:
        # Parse original
        tree = ET.parse(orig_file)
        root = tree.getroot()
        
        # Modify all flow elements: change end from 3600 to 1000, keep perHour unchanged
        flows_modified = 0
        for flow in root.findall('.//flow'):
            if flow.get('end') == '3600':
                flow.set('end', '1000')
                flows_modified += 1
        
        # Save as new file with _t1000 suffix
        new_name = orig_file.stem + "_t1000" + orig_file.suffix
        new_file = base_dir / new_name
        tree.write(new_file, encoding='UTF-8', xml_declaration=True)
        print(f"  Created {new_name} ({flows_modified} flows modified)")

if __name__ == "__main__":
    for demand in [500, 750, 1000]:
        create_t1000_routes(demand)
    print("\nDone! Created corrected T1000 route files.")
