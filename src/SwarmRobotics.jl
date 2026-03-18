# src/SwarmRobotics.jl
module SwarmRobotics

# 1. Import all dependencies for the entire package
using AbstractGPs
using KernelFunctions
using Random
using LinearAlgebra
using MeshCat
using GeometryBasics
using Colors
using CoordinateTransformations

# 2. Export the function we are about to create so users can call it
export run_active_mapping

# 3. Include your logic script
include("spatial_mapping.jl")

end