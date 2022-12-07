#!/usr/bin/env python
"""
Reads a hydro particleset and creates a new particleset based on the density
field.
Arguments:
 - original particle set
 - factor with which to multiply the original particle number
"""
import sys

import numpy as np
from amuse.community.fi import Fi
from amuse.datamodel import Particles
from amuse.io import read_set_from_file
from amuse.io import write_set_to_file
from amuse.units import nbody_system
from amuse.units import units
from numpy.random import default_rng
from numpy.random import Generator
from numpy.random import random_sample
from scipy.interpolate import RegularGridInterpolator

DENSITY_UNIT = units.g * units.cm**-3
LENGTH_UNIT = units.pc
MASS_UNIT = units.MSun
TIME_UNIT = units.Myr

np.random.seed(5)  # for reproducibility


def random_guess_near_orig(number_of_points, original_points, **kwargs):
    rng = default_rng()
    valid_x = np.zeros(number_of_points) | units.m
    valid_y = np.zeros(number_of_points) | units.m
    valid_z = np.zeros(number_of_points) | units.m
    particles_chosen = original_points[rng.integers(0,
                                                    high=len(original_points),
                                                    size=number_of_points)]
    valid_x = particles_chosen.x + (
        (rng.random(number_of_points) - 0.5) * particles_chosen.h_smooth)
    valid_y = particles_chosen.y + (
        (rng.random(number_of_points) - 0.5) * particles_chosen.h_smooth)
    valid_z = particles_chosen.z + (
        (rng.random(number_of_points) - 0.5) * particles_chosen.h_smooth)
    return (
        valid_x,
        valid_y,
        valid_z,
    )


def random_pdf_3d(
    number_of_points,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    rhomax,
    pdf,
    oversample=1,
    **kwargs,
):
    sample_size = int(number_of_points * oversample)
    print(f"Sample size is {sample_size}")
    valid_x = np.zeros(number_of_points) | units.m
    valid_y = np.zeros(number_of_points) | units.m
    valid_z = np.zeros(number_of_points) | units.m
    print("Looping")
    number_done = 0
    while number_done < number_of_points:
        print(f"Found {number_done} valid points out of {number_of_points}")
        x_choice = xmin + (xmax - xmin) * random_sample(sample_size)
        y_choice = ymin + (ymax - ymin) * random_sample(sample_size)
        z_choice = zmin + (zmax - zmin) * random_sample(sample_size)
        rho_choice = random_sample(sample_size) * rhomax
        rho_answer = pdf(x_choice, y_choice, z_choice, **kwargs)
        valid_answers = np.where(rho_choice <= rho_answer)
        number_done_new = min(number_of_points - number_done,
                              len(valid_answers[0]))
        valid_x[number_done:number_done +
                number_done_new] = x_choice[valid_answers[0]][:number_done_new]
        valid_y[number_done:number_done +
                number_done_new] = y_choice[valid_answers[0]][:number_done_new]
        valid_z[number_done:number_done +
                number_done_new] = z_choice[valid_answers[0]][:number_done_new]
        number_done += number_done_new
    return (
        valid_x[:number_of_points],
        valid_y[:number_of_points],
        valid_z[:number_of_points],
    )


def create_hydro_instance(
    particles,
    **kwargs,
):
    """
    Returns a hydro instance with the best guess of converter settings.
    """
    field_code = kwargs.get("field_code") or Fi
    converter = kwargs.pop("convert_nbody") or nbody_system.nbody_to_si(
        particles.total_mass(),
        100 | units.pc,
    )
    kwargs["redirection"] = "none"
    instance = field_code(converter, **kwargs)
    instance.parameters.timestep = 1 | units.s
    if hasattr(particles, "itype"):
        particles = particles[particles.itype == 1]
        del particles.itype
    instance.gas_particles.add_particles(particles)
    return instance


def get_density(*args, **kwargs):
    """
    Returns the density term from the hydro state.
    """
    instance = kwargs.get("instance")
    if instance is None:
        field_particles = kwargs.get("field_particles")
        if field_particles is None:
            return -1
        instance = create_hydro_instance(field_particles, *args, **kwargs)
    fields = instance.get_hydro_state_at_point(*args, **kwargs)
    return fields[0]


def get_pressure(*args, **kwargs):
    """
    Calculates pressure term from hydro state.
    """
    gamma = kwargs.get("gamma") or 1.0
    instance = kwargs.get("instance")
    if instance is None:
        field_particles = kwargs.get("field_particles")
        if field_particles is None:
            return -1
        instance = create_hydro_instance(field_particles, *args, **kwargs)
    fields = instance.get_hydro_state_at_point(*args, **kwargs)
    rho, rho_vx, rho_vy, rho_vz, rho_e = fields

    specific_energy = rho_e - 0.5 * (rho_vx**2 + rho_vy**2 + rho_vz**2) / rho
    pressure_term = gamma * rho * specific_energy
    return pressure_term


def main():
    """
    Returns a resampling of hydro particles using the pressure term to
    determine new particle locations.
    """
    print("Reading initial file")
    particles_original = read_set_from_file(sys.argv[1])  # Particles()
    if hasattr(particles_original, "itype"):
        particles_original = particles_original[particles_original.itype == 1]
    xmin = particles_original.x >= -2100 | units.pc
    p = particles_original[xmin]
    xmax = p.x <= -1500 | units.pc
    p = p[xmax]
    ymin = p.y >= -2100 | units.pc
    p = p[ymin]
    ymax = p.y <= -1500 | units.pc
    particles_original = p[ymax]
    # converter = nbody_system.nbody_to_si(
    #     particles_original.total_mass(), 1 | units.pc
    # )
    print(f"Creating {int(len(particles_original) * float(sys.argv[2]))} "
          "new particles")
    particles_new = Particles(int(
        len(particles_original) * float(sys.argv[2])))

    boundary_x_min = particles_original.x.min()
    boundary_y_min = particles_original.y.min()
    boundary_z_min = particles_original.z.min()
    boundary_x_max = particles_original.x.max()
    boundary_y_max = particles_original.y.max()
    boundary_z_max = particles_original.z.max()

    # Make an initial guess for the new particles' positions.
    # This could be:
    # - a grid (simple, takes more iterations)
    # - positions close to the original particles (only when doing an integer
    #   times the original number)
    #   offset these positions (x, y and z) by a random number ((0, 1]) times
    #   the smoothing length
    # - positions based on the original density field, using this as a PDF to
    #   generate random positions (maybe the best option, but not the easiest)
    #   - get PDF (= density on a grid of points, interpolated with
    #     scipy.interpolate.RegularGridInterpolator)
    #   - normalise PDF (?)
    #   - use PDF to generate N random samples

    print("Creating SPH instance")
    sph_original_instance = create_hydro_instance(
        particles_original,
        convert_nbody=None,
        mode="openmp",
    )
    print("Copying density and smoothing length")
    sph_original_instance.gas_particles.new_channel_to(
        particles_original).copy_attributes(["h_smooth", "rho"])
    write_set_to_file(sph_original_instance.gas_particles,
                      "gas-original.amuse",
                      overwrite_file=True)
    print("Getting max density")
    # This should work but gives weird results! FIXME
    # rho_max = sph_original_instance.get_rhomax()
    print(particles_original[0])
    print(sph_original_instance.gas_particles[0])
    rho_max = particles_original.rho.max()
    print(f"Max density is {rho_max}")

    print("Creating a population guess")
    kwargs = {}
    kwargs["instance"] = sph_original_instance

    positions_guess = random_guess_near_orig(
        len(particles_new),
        particles_original,
    )
    # positions_guess = random_pdf_3d(
    #     len(particles_new),
    #     boundary_x_min,
    #     boundary_x_max,
    #     boundary_y_min,
    #     boundary_y_max,
    #     boundary_z_min,
    #     boundary_z_max,
    #     rho_max,
    #     get_density,
    #     oversample=10,
    #     **kwargs
    # )
    particles_new.x = positions_guess[0]
    particles_new.y = positions_guess[1]
    particles_new.z = positions_guess[2]
    particles_new.mass = particles_original.total_mass() / len(particles_new)

    fields = sph_original_instance.get_hydro_state_at_point(
        particles_new.x, particles_new.y, particles_new.z)
    particles_new.vx = fields[1] / fields[0]
    particles_new.vy = fields[2] / fields[0]
    particles_new.vz = fields[3] / fields[0]
    particles_new.u = fields[4] / fields[0]

    sph_new_instance = create_hydro_instance(particles_new,
                                             **{"convert_nbody": None})
    write_set_to_file(sph_new_instance.gas_particles,
                      "gas-guess.amuse",
                      overwrite_file=True)

    number_of_iterations = 10
    for i in range(number_of_iterations):
        print(f"Iteration {i+1}/{number_of_iterations}")
        sph_new_instance.gas_particles.new_channel_to(particles_new).copy()
        kwargs = {}
        small_length = 1 * (sph_new_instance.gas_particles.h_smooth.min()
                            ).value_in(LENGTH_UNIT)
        pressure_orig = []
        pressure_new = []
        pressure_difference = []
        for offset in [
            [-small_length, 0, 0] | LENGTH_UNIT,
            [small_length, 0, 0] | LENGTH_UNIT,
            [0, -small_length, 0] | LENGTH_UNIT,
            [0, small_length, 0] | LENGTH_UNIT,
            [0, 0, -small_length] | LENGTH_UNIT,
            [0, 0, small_length] | LENGTH_UNIT,
        ]:
            kwargs["instance"] = sph_original_instance
            pressure_orig.append(
                get_pressure(
                    particles_new.x + offset[0],
                    particles_new.y + offset[1],
                    particles_new.z + offset[2],
                    **kwargs,
                )**0.5)
            print("***** orig *****")
            kwargs["instance"] = sph_new_instance
            pressure_new.append(
                get_pressure(
                    particles_new.x + offset[0],
                    particles_new.y + offset[1],
                    particles_new.z + offset[2],
                    **kwargs,
                )**0.5)
            print("***** new  *****")
            pressure_difference.append(
                pressure_orig[-1] - pressure_new[-1])  # / particles_new.mass

        pressure_difference_gradient_dx = (pressure_difference[1] -
                                           pressure_difference[0]) / (
                                               2 * small_length | LENGTH_UNIT)
        pressure_difference_gradient_dy = (pressure_difference[3] -
                                           pressure_difference[2]) / (
                                               2 * small_length | LENGTH_UNIT)
        pressure_difference_gradient_dz = (pressure_difference[5] -
                                           pressure_difference[4]) / (
                                               2 * small_length | LENGTH_UNIT)

        PRESSURE_UNIT = units.MSun * units.pc**-2 * units.Myr**-1
        print(
            pressure_difference_gradient_dx[0].in_(PRESSURE_UNIT *
                                                   units.pc**-1),
            pressure_difference_gradient_dy[0].in_(PRESSURE_UNIT *
                                                   units.pc**-1),
            pressure_difference_gradient_dz[0].in_(PRESSURE_UNIT *
                                                   units.pc**-1),
        )
        pdmax = max(
            max(
                max(pressure_difference_gradient_dx),
                -min(pressure_difference_gradient_dx),
            ),
            max(
                max(pressure_difference_gradient_dy),
                -min(pressure_difference_gradient_dy),
            ),
            max(
                max(pressure_difference_gradient_dz),
                -min(pressure_difference_gradient_dz),
            ),
        )
        pdmove_x = pressure_difference_gradient_dx / pdmax
        pdmove_y = pressure_difference_gradient_dy / pdmax
        pdmove_z = pressure_difference_gradient_dz / pdmax

        particles_new.x -= (pdmove_x * (1 - i / number_of_iterations) *
                            particles_new.h_smooth)
        # (
        #     pressure_difference_gradient_dx.value_in(
        #         PRESSURE_UNIT * units.pc**-1
        #     ) | 100*units.pc
        # )
        particles_new.y -= (pdmove_y * (1 - i / number_of_iterations) *
                            particles_new.h_smooth)
        # (
        #     pressure_difference_gradient_dy.value_in(
        #         PRESSURE_UNIT * units.pc**-1
        #     ) | 100*units.pc
        # )
        particles_new.z -= (pdmove_z * (1 - i / number_of_iterations) *
                            particles_new.h_smooth)
        # (
        #     pressure_difference_gradient_dz.value_in(
        #         PRESSURE_UNIT * units.pc**-1
        #     ) | 100*units.pc
        # )
        particles_new.new_channel_to(
            sph_new_instance.gas_particles).copy_attributes(["x", "y", "z"])
        print(sph_new_instance.gas_particles[0].position)

        write_set_to_file(sph_new_instance.gas_particles,
                          f"gas-iter-{i+1}.amuse",
                          overwrite_file=True)


#
#     length_step_max = particles_original.h_smooth.max()
#     length_step_min = particles_original.h_smooth.min()
#     length_step_steps = 10
#     length_steps = np.logspace(
#         np.log10(length_step_min.value_in(LENGTH_UNIT)),
#         np.log10(length_step_max.value_in(LENGTH_UNIT)),
#         length_step_steps
#     ) | LENGTH_UNIT
#
#     for length_step in length_steps:
#         x_step_number = int(
#             (boundary_x_max - boundary_x_min) / length_step
#         )
#         y_step_number = int(
#             (boundary_y_max - boundary_y_min) / length_step
#         )
#         z_step_number = int(
#             (boundary_z_max - boundary_z_min) / length_step
#         )
#
#         # meshgrid_array = np.meshgrid(
#         x = np.linspace(
#             (boundary_x_min).value_in(LENGTH_UNIT),
#             (boundary_x_max).value_in(LENGTH_UNIT),
#             x_step_number+2
#         )
#         y = np.linspace(
#             (boundary_y_min).value_in(LENGTH_UNIT),
#             (boundary_y_max).value_in(LENGTH_UNIT),
#             y_step_number+2
#         )
#         z = np.linspace(
#             (boundary_z_min).value_in(LENGTH_UNIT),
#             (boundary_z_max).value_in(LENGTH_UNIT),
#             z_step_number+2
#         )
#         xgrid, ygrid, zgrid = np.meshgrid(x, y, z, indexing='ij') | LENGTH_UNIT
#         particles_grid = Particles(len(x) * len(y) * len(z))
#         particles_grid.x = xgrid.flatten()
#         particles_grid.y = ygrid.flatten()
#         particles_grid.z = zgrid.flatten()
#
#         # rho, rhovx, rhovy, rhovz, rhoe
#         densitystate_original = sph_original.get_hydro_state_at_point(
#             particles_grid.position)
#         rho_original = densitystate_original[0]
#         density_interpolator = RegularGridInterpolator(
#             (x, y, z),
#             rho_original.value_in(DENSITY_UNIT)
#         )
#
#         guess_x, guess_y, guess_z = random_pdf_3d(
#             len(particles_new),
#             boundary_x_min, boundary_x_min,
#             boundary_y_min, boundary_y_min,
#             boundary_z_min, boundary_z_min,
#             particles_original.rho.max(),
#             density_interpolator
#         )
#         particles_new.x = guess_x
#         particles_new.y = guess_y
#         particles_new.z = guess_z
#
#         densitystate_new = get_hydro_fields(
#             particles_new,
#             query_locations=particles_grid.position,
#             instance=None,
#             field_generating_code=Fi,
#             default_settings={
#                 'convert_nbody': nbody_system.nbody_to_si(
#                     1 | units.MSun, 0.1 | units.pc),
#                 'mode': "openmp",
#             },
#             return_instance=False,
#         )
#         # sph_new.get_hydro_state_at_point(
#         #     particles_grid.position)
#
#         rho_new = densitystate_new[0]
#
#         rho_difference = rho_new - rho_original
#         rho_difference.reshape(
#             (x_step_number+2, y_step_number+2, z_step_number+2)
#         )
#
#         drhodx_gradient = np.gradient(
#             rho_difference.value_in(DENSITY_UNIT), axis=0
#         ) | DENSITY_UNIT / length_step.as_unit()
#         drhody_gradient = np.gradient(
#             rho_difference.value_in(DENSITY_UNIT), axis=1
#         ) | DENSITY_UNIT / length_step.as_unit()
#         drhodz_gradient = np.gradient(
#             rho_difference.value_in(DENSITY_UNIT), axis=2
#         ) | DENSITY_UNIT / length_step.as_unit()
#
#
#     # timestep =

if __name__ == "__main__":
    main()
