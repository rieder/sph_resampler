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
from numpy.random import random_sample
from scipy.interpolate import RegularGridInterpolator
from amuse.io import read_set_from_file, write_set_to_file
from amuse.units import units, nbody_system
from amuse.datamodel import Particles
from amuse.community.fi import Fi


DENSITY_UNIT = units.g * units.cm**-3
LENGTH_UNIT = units.pc
MASS_UNIT = units.MSun
TIME_UNIT = units.Myr


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
    oversample=10,
):
    sample_size = int(number_of_points * oversample)
    valid_x = []
    valid_y = []
    valid_z = []
    while len(valid_x) < number_of_points:
        x_choice = xmin + (xmax - xmin) * random_sample(sample_size)
        y_choice = ymin + (ymax - ymin) * random_sample(sample_size)
        z_choice = zmin + (zmax - zmin) * random_sample(sample_size)
        rho_choice = random_sample(sample_size) * rhomax
        rho_answer = pdf(x_choice, y_choice, z_choice)
        valid_answers = np.where(
            rho_choice <= rho_answer
        )
        valid_x.append(x_choice[valid_answers])
        valid_y.append(x_choice[valid_answers])
        valid_z.append(x_choice[valid_answers])
    return (
        valid_x[:number_of_points],
        valid_y[:number_of_points],
        valid_z[:number_of_points],
    )


def get_hydro_fields(
    particles_creating_field,
    query_locations=[[0, 0, 0]] | units.pc,
    instance=None,
    field_generating_code=Fi,
    default_settings={
        'convert_nbody': nbody_system.nbody_to_si(
            1 | units.MSun, 0.1 | units.pc),
    },
    return_instance=False,
):
    """
    Sets up a hydro instance and returns the hydro state or the instance
    itself.
    """
    if instance is None:
        if default_settings.get('mode') is None:
            default_settings['mode'] = "openmp"
        instance = field_generating_code(**default_settings)
        instance.parameters.timestep = 1 | units.s
        instance.gas_particles.add_particles(particles_creating_field)
    if return_instance:
        return instance
    fields = instance.get_hydro_state_at_point(query_locations)
    return fields


def get_pressure(
    *args, **kwargs
):
    """
    Calculates pressure term from hydro state.
    """
    gamma = kwargs.get('gamma') or 1.0
    return_instance = kwargs.get('return_instance')
    fields = get_hydro_fields(*args, **kwargs)
    if return_instance is not None:
        return fields
    rho, rho_vx, rho_vy, rho_vz, rho_e = fields

    specific_energy = rho_e - 0.5 * (rho_vx**2 + rho_vy**2 + rho_vz**2) / rho
    pressure_term = gamma * rho * specific_energy
    return pressure_term


def main():
    """
    Returns a resampling of hydro particles using the pressure term to
    determine new particle locations.
    """
    particles_original = read_set_from_file(sys.argv[1])  # Particles()
    converter = nbody_system.nbody_to_si(
        particles_original.total_mass(), 1 | units.pc
    )
    particles_new = Particles(
        int(len(particles_original) * float(sys.argv[2]))
    )

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

    sph_original_instance = get_hydro_fields(
        particles_original, return_instance=True
    )
    sph_original_instance.gas_particles.new_channel_to(
        particles_original
    ).copy_attributes(["h_smooth", "rho"])

    length_step_max = particles_original.h_smooth.max()
    length_step_min = particles_original.h_smooth.min()
    length_step_steps = 10
    length_steps = np.logspace(
        np.log10(length_step_min.value_in(LENGTH_UNIT)),
        np.log10(length_step_max.value_in(LENGTH_UNIT)),
        length_step_steps
    ) | LENGTH_UNIT

    for length_step in length_steps:
        x_step_number = int(
            (boundary_x_max - boundary_x_min) / length_step
        )
        y_step_number = int(
            (boundary_y_max - boundary_y_min) / length_step
        )
        z_step_number = int(
            (boundary_z_max - boundary_z_min) / length_step
        )

        # meshgrid_array = np.meshgrid(
        x = np.linspace(
            (boundary_x_min).value_in(LENGTH_UNIT),
            (boundary_x_max).value_in(LENGTH_UNIT),
            x_step_number+2
        )
        y = np.linspace(
            (boundary_y_min).value_in(LENGTH_UNIT),
            (boundary_y_max).value_in(LENGTH_UNIT),
            y_step_number+2
        )
        z = np.linspace(
            (boundary_z_min).value_in(LENGTH_UNIT),
            (boundary_z_max).value_in(LENGTH_UNIT),
            z_step_number+2
        )
        xgrid, ygrid, zgrid = np.meshgrid(x, y, z, indexing='ij') | LENGTH_UNIT
        particles_grid = Particles(len(x) * len(y) * len(z))
        particles_grid.x = xgrid.flatten()
        particles_grid.y = ygrid.flatten()
        particles_grid.z = zgrid.flatten()

        # rho, rhovx, rhovy, rhovz, rhoe
        densitystate_original = sph_original.get_hydro_state_at_point(
            particles_grid.position)
        rho_original = densitystate_original[0]
        density_interpolator = RegularGridInterpolator(
            (x, y, z),
            rho_original.value_in(DENSITY_UNIT)
        )

        guess_x, guess_y, guess_z = random_pdf_3d(
            len(particles_new),
            boundary_x_min, boundary_x_min,
            boundary_y_min, boundary_y_min,
            boundary_z_min, boundary_z_min,
            particles_original.rho.max(),
            density_interpolator
        )
        particles_new.x = guess_x
        particles_new.y = guess_y
        particles_new.z = guess_z

        densitystate_new = get_hydro_fields(
            particles_new,
            query_locations=particles_grid.position,
            instance=None,
            field_generating_code=Fi,
            default_settings={
                'convert_nbody': nbody_system.nbody_to_si(
                    1 | units.MSun, 0.1 | units.pc),
                'mode': "openmp",
            },
            return_instance=False,
        )
        # sph_new.get_hydro_state_at_point(
        #     particles_grid.position)

        rho_new = densitystate_new[0]

        rho_difference = rho_new - rho_original
        rho_difference.reshape(
            (x_step_number+2, y_step_number+2, z_step_number+2)
        )

        drhodx_gradient = np.gradient(
            rho_difference.value_in(DENSITY_UNIT), axis=0
        ) | DENSITY_UNIT / length_step.as_unit()
        drhody_gradient = np.gradient(
            rho_difference.value_in(DENSITY_UNIT), axis=1
        ) | DENSITY_UNIT / length_step.as_unit()
        drhodz_gradient = np.gradient(
            rho_difference.value_in(DENSITY_UNIT), axis=2
        ) | DENSITY_UNIT / length_step.as_unit()


    # timestep = 


if __name__ == "__main__":
    main()
