# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:39:11 2025

@author: bilou
"""

# XACRO to URDF CONVERSION

import xacro_to_urdf

xacro_path = r"/mnt/c/Users/bilou/OneDrive/Documents/Ecole/MASTER 2/TFE/ShadowHand/urdf/sr_hand.urdf.xacro"
urdf_path = r"C:\Users\bilou\OneDrive\Documents\Ecole\MASTER 2\TFE\ShadowHand\urdf\sr_hand.urdf"
parameters = {
    "side": "right",
    "hand_type": "hand_e",
    "hand_version": "E3M5",
    "fingers": "all"
}

# Run the conversion directly
# xacro_to_urdf.convert_xacro_to_urdf(xacro_path, urdf_path, parameters)

# URDF to MJCF CONVERSION

import xml.etree.ElementTree as ET
import mujoco

def convert_urdf_to_mjcf(urdf_path, mjcf_path):
    # Parse the URDF XML
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Create the MJCF XML structure
    mjcf_root = ET.Element('mujoco', model='robot')

    # Example: Convert URDF links and joints to MJCF equivalents
    for link in root.findall('link'):
        mjcf_link = ET.SubElement(mjcf_root, 'body', name=link.get('name'))

        # Example: add inertial properties (mass, inertia, etc.)
        inertial = link.find('inertial')
        if inertial is not None:
            mass = inertial.find('mass')
            if mass is not None:
                # Handle None case by checking for missing values and assigning defaults
                mass_value = mass.text if mass.text is not None else '0'
                ET.SubElement(mjcf_link, 'inertial', mass=mass_value)
            else:
                # If mass is None, assign a default value or skip
                ET.SubElement(mjcf_link, 'inertial', mass='0')

    # Write the MJCF XML to a file
    tree_mjcf = ET.ElementTree(mjcf_root)
    
    # Ensure we handle any missing elements or attributes before writing
    try:
        tree_mjcf.write(mjcf_path, encoding="utf-8", xml_declaration=True)
        print("Conversion to MJCF successful!")
    except Exception as e:
        print(f"Error during MJCF conversion: {e}")

# Example usage
mjcf_path = r"C:\Users\bilou\OneDrive\Documents\Ecole\MASTER 2\TFE\ShadowHand\urdf\sr_hand.xml"
convert_urdf_to_mjcf(urdf_path, mjcf_path)
