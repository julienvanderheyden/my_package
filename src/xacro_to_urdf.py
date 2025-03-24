# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:50:35 2025

@author: bilou
"""

import xacro
import argparse



def convert_xacro_to_urdf(xacro_path, urdf_path, parameters):
    """
    Convert a Xacro file to a URDF file with customizable parameters.

    :param xacro_path: Path to the input Xacro file.
    :param urdf_path: Path to save the output URDF file.
    :param parameters: Dictionary of parameters to pass to Xacro.
    """
    # Process the Xacro file with given parameters
    urdf_str = xacro.process_file(xacro_path, mappings=parameters).toxml()

    # Save the generated URDF
    with open(urdf_path, "w") as f:
        f.write(urdf_str)

    print(f"Xacro successfully converted to URDF! Saved at: {urdf_path}")

if __name__ == "__main__":
    # Argument parser for command-line use
    parser = argparse.ArgumentParser(description="Convert a Xacro file to a URDF file with parameters.")

    parser.add_argument("xacro_path", type=str, help="Path to the input Xacro file.")
    parser.add_argument("urdf_path", type=str, help="Path to save the output URDF file.")
    
    # Optional key-value pairs for Xacro parameters
    parser.add_argument("--param", action="append", nargs=2, metavar=("key", "value"),
                        help="Specify parameters to pass to the Xacro file (e.g., --param wheel_radius 0.15)")

    args = parser.parse_args()

    # Convert list of key-value pairs into a dictionary
    param_dict = {key: value for key, value in args.param} if args.param else {}

    # Convert the Xacro file
    convert_xacro_to_urdf(args.xacro_path, args.urdf_path, param_dict)
