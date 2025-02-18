'''Aron Jacobse & Juho Kauppi'''

from nutils import mesh, function, solver, export, cli
from nutils.expression_v2 import Namespace
import treelog
import matplotlib.pyplot as plt
import numpy as np


# === Dictionary with user input ===
data = {

    # Starting variable should always be defined
    "starting variables" : {
        "length": 0.1,  # Length of the geometry [m]
        "width": 0.1,  # Width of the geometry [m]
        "height": 0.01,  # Height of the geometry [m]
        "elements X": 10,  # Number of elements in x-direction (length) [-]
        "elements Y": 10,  # Number of elements in y-direction (width) [-]
        "elements Z": 5,  # Number of elements in z-direction (height) [-]
        "T air": 25,  # Ambient air temperature [°C]
        "convective coefficient": 10,  # Convective heat transfer coefficient [W/m²·K]
        "thermal expansion coefficient": 1.2e-5,  # Coefficient of thermal expansion [1/K]
        "young's modulus": 210e9,  # Young's modulus of the material [Pa]
        "poisson's ratio": 0.3  # Poisson's ratio of the material [-]
    },

    # The number of boundaries can be chosen freely, ranging from 0 to infinity
    # All different types of possible boundaries have been applied in this example
    "boundaries": {

        "boundary1": {
            "solver": "thermal",  # Stating the solver, thermal or elastic
            "side": "top",  # Stating the side where to apply the boundary on, with as possibilities the 6 sides of a cube
            "type": "neumann",  # Stating what kind of boundary, Dirichlet or Neumann or Convective cooling
            "value": 10  # Stating the value to be applied
        },

        "boundary2": {
            "solver": "elastic",
            "side": "bottom",
            "type": "dirichlet",
            "value": [0, 0, 0]  # This boundary corresponds to a displacement, so displacement in X, Y and Z direction should be supplied
        },

        "boundary3": {
            "solver": "elastic",
            "side": "left",
            "type": "neumann",
            "value": 5,  # This boundary corresponds to a force, here the magnitude should be supplied
            "direction": [0.0, 0.0, -1.0]  # Here the force direction should be supplied in form of a unit vector
        },

        "boundary4": {
            "solver": "thermal",
            "side": "left",
            "type": "dirichlet",
            "value": 6
        },

        "boundary5": {
            "solver": "thermal",
            "side": "bottom",
            "type": "cooling",
            "value": 7
        },

        "boundary6": {
            "solver": "elastic",
            "side": "right",
            "type": "dirichlet",
            "value": [0.1, 0.01, 0.02]
        }
    }
}

def main():

    # Reading in the starting parameters from the user input
    starting_variables = data["starting variables"]

    # === Simulation Parameters ===
    length, width, height = starting_variables["length"], starting_variables["width"], starting_variables["height"]  # Dimensions of the domain (meters)
    elements_x, elements_y, elements_z = starting_variables["elements X"], starting_variables["elements Y"], starting_variables["elements Z"]  # Reading in the amount of elements in every direction
    t_air = starting_variables["T air"]  # Ambient air temperature (°C)
    conv_coeff = starting_variables["convective coefficient"]  # Convective heat transfer coefficient (W/m²·K)
    alpha = starting_variables["thermal expansion coefficient"]  # Coefficient of thermal expansion (1/K)
    youngs_modulus = starting_variables["young's modulus"]  # Young's modulus of the material (Pa)
    nu = starting_variables["poisson's ratio"]  # Poisson's ratio of the material (dimensionless)

    # === Mesh Creation ===
    domain, geom = mesh.rectilinear([np.linspace(0, length, elements_x + 1),  # X-coordinates
                                    np.linspace(0, width, elements_y + 1),   # Y-coordinates
                                    np.linspace(0, height, elements_z + 1)])  # Z-coordinates

    # === Namespace Setup ===
    # The namespace is used to define symbolic expressions and functions within the Nutils framework.
    ns = Namespace()
    ns.x = geom
    ns.define_for('x', gradient='∇', normal='n', jacobians=('dV', 'dS'))
    ns.δ = function.eye(domain.ndims)

    # === Thermal Problem Setup ===
    # Add a temperature field 'T' and its test function 'v'
    ns.add_field(('T', 'v'), domain.basis('std', degree=1))
    ns.h = conv_coeff

    # Define the weak form of the thermal problem
    thermal_res = domain.integral('∇_i(v) ∇_i(T) dV' @ ns, degree=2)

    # Reading in the boundaries from the input
    boundaries = data['boundaries']

    # === Apply Thermal Boundary Conditions ===
    for boundary_name, b_thermal in boundaries.items():

        if b_thermal['solver'] == "thermal":

            if b_thermal["type"] == "dirichlet":
                ns.T0 = b_thermal["value"]

                thermal_res -= domain.boundary[b_thermal["side"]].integral('(T - T0)^2 dS' @ ns, degree=2)

            elif b_thermal["type"] == "neumann":
                ns.qtop = b_thermal["value"]

                thermal_res -= domain.boundary[b_thermal["side"]].integral('v qtop dS' @ ns, degree=2)

            elif b_thermal["type"] == "cooling":
                ns.Tair = b_thermal["value"]

                thermal_res += domain.boundary[b_thermal["side"]].integral('v h (T - Tair) dS' @ ns, degree=2)

    # === Solve the Thermal Problem ===
    # The temperature field 'T' is computed using a linear solver.
    thermal_args = solver.solve_linear('T:v', thermal_res)

    # === Elasticity Problem Setup ===
    # Add a displacement field 'u' and its test function 'w'
    ns.add_field(('u', 'w'), domain.basis('std', degree=1), shape=(3,))
    ns.λ = youngs_modulus * nu / ((1 + nu) * (1 - 2 * nu))
    ns.μ = youngs_modulus / (2 * (1 + nu))
    ns.Tref = t_air
    ns.alpha = alpha

    # Define the stress and strain tensor, including the thermal strain component
    ns.ε_ij = '.5 (∇_i(u_j) + ∇_j(u_i)) - alpha (T - Tref) δ_ij'
    ns.σ_ij = 'λ ε_kk δ_ij + 2 μ ε_ij'

    # Define the weak form of the elasticity problem (Linear momentum balance equation)
    res = domain.integral('∇_j(w_i) σ_ij dV' @ ns, degree=2)

    # === Apply Elastic Boundary Conditions ===
    for boundary_name, b_elastic in boundaries.items():

        if b_elastic["solver"] == "elastic":

            if b_elastic["type"] == "dirichlet":
                displacement = b_elastic["value"]
                ns.g = np.array(displacement)
                sqr = domain.boundary[b_elastic["side"]].integral('(u_k - g_k) (u_k - g_k) dS' @ ns, degree=2)
                cons = solver.optimize('u,', sqr, droptol=1e-15)

            elif b_elastic["type"] == "neumann":
                direction = b_elastic["direction"]
                factor = b_elastic["value"]
                scaled_list = [v*factor for v in direction]
                ns.t = np.array(scaled_list)

                res += domain.boundary[b_elastic["side"]].integral('t_k w_k dS' @ ns, degree=2)

    # Solve elasticity problem
    elastic_args = solver.solve_linear('u:w', res, constrain=cons, arguments={**thermal_args, **{'T': thermal_args['T']}})

    # === Merge Thermal and Elastic Results ===
    # This ensures compatibility when visualizing and exporting data.
    all_args = {**thermal_args, **elastic_args}

    # === Visualization: Export to VTK ===
    # Sample a Bezier grid for smooth visualization
    bezier = domain.sample('bezier', 10)

    # Evaluate fields at the Bezier sample points
    x_vals = bezier.eval(ns.x, **all_args)  # Original coordinates
    u_vals = bezier.eval(ns.u, **all_args)  # Displacement field
    T_vals = bezier.eval(ns.T, **thermal_args)  # Temperature field

    # Compute displacement magnitude
    u_magnitude = np.sqrt((u_vals**2).sum(axis=1))

    # Plot displacement magnitude
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], c=u_magnitude, cmap='viridis'  # Scatter plot of displacement magnitude
    )

    # Set aspect ratio to equal for the real-world geometry
    ax.set_box_aspect([length, width, height])
    plt.colorbar(sc, label='Displacement Magnitude', shrink=0.6, aspect=20, pad=0.15)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Displacement Magnitude (m)')
    plt.savefig('displacement_magnitude.png')
    plt.show()

    # Plot temperature distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], c=T_vals, cmap='plasma'
    )
    # Set aspect ratio to equal for the real-world geometry
    ax.set_box_aspect([length, width, height])  # Match the real dimensions
    plt.colorbar(sc, label='Temperature (°C)', shrink=0.6, aspect=20, pad=0.15)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Temperature Distribution')
    plt.savefig('temperature_distribution.png')
    plt.show()

    # Export to VTK
    # export.vtk(
    #     'results',  # Name of the output file (without extension)
    #     bezier.tri,  # Converted triangle connectivity
    #     x_vals,  # Original coordinates
    #     T=T_vals,  # Temperature field
    #     u=u_vals  # Displacement field
    # )

    treelog.info("Deformed geometry and fields exported to 'deformed_geometry.vtk'.")

    return

if __name__ == '__main__':
    cli.run(main)
