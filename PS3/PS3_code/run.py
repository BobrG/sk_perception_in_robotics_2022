#!/usr/bin/python

"""
Sudhanva Sreesha
ssreesha@umich.edu
22-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
28-February-2021
"""

import contextlib
import os
from argparse import ArgumentParser

import numpy as np
from scipy.sparse.linalg import inv
from matplotlib import pyplot as plt
from progress.bar import FillingCirclesBar

from tools.objects import Gaussian
from tools.plot import get_plots_figure
from tools.plot import plot_robot
from field_map import FieldMap
#from slam import SimulationSlamBase
from tools.data import generate_data as generate_input_data
from tools.data import load_data
from tools.plot import plot_field
from tools.plot import plot_observations
from tools.task import get_dummy_context_mgr
from tools.task import get_movie_writer
from tools.plot import plot2dcov 

from slam.sam import Sam

import mrob

def get_cli_args():
    parser = ArgumentParser('Perception in Robotics PS3')
    parser.add_argument('-i',
                        '--input-data-file',
                        type=str,
                        action='store',
                        help='File with generated data to simulate the filter '
                             'against. Supported format: "npy", and "mat".')
    parser.add_argument('-n',
                        '--num-steps',
                        type=int,
                        action='store',
                        help='The number of time steps to generate data for the simulation. '
                             'This option overrides the data file argument.',
                        default=100)
    parser.add_argument('-f',
                        '--filter',
                        dest='filter_name',
                        choices=['ekf', 'sam'],
                        action='store',
                        help='The slam filter use for the SLAM problem.',
                        default='sam')
    parser.add_argument('-a',
                        '--alphas',
                        type=float,
                        nargs=4,
                        metavar=('A1', 'A2', 'A3', 'A4'),
                        action='store',
                        help='Diagonal of Standard deviations of the Transition noise in action space (M_t).',
                        default=(0.05, 0.001, 0.05, 0.01))
    parser.add_argument('-b',
                        '--beta',
                        type=float,
                        nargs=2,
                        metavar=('range', 'bearing (deg)'),
                        action='store',
                        help='Diagonal of Standard deviations of the Observation noise (Q). (format: cm deg).',
                        default=(10., 10.))
    parser.add_argument('--dt', type=float, action='store', help='Time step (in seconds).', default=0.1)
    parser.add_argument('-s', '--animate', action='store_true', help='Show and animation of the simulation, in real-time.')
    parser.add_argument('--plot-pause-len',
                        type=float,
                        action='store',
                        help='Time (in seconds) to pause the plot animation for between frames.',
                        default=0.01)
    parser.add_argument('--num-landmarks-per-side',
                        type=int,
                        help='The number of landmarks to generate on one side of the field.',
                        default=4)
    parser.add_argument('--max-obs-per-time-step',
                        type=int,
                        help='The maximum number of observations to generate per time step.',
                        default=2)
    parser.add_argument('--data-association',
                        type=str,
                        choices=['known', 'ml', 'jcbb'],
                        default='known',
                        help='The type of data association algorithm to use during the update step.')
    parser.add_argument('--update-type',
                        type=str,
                        choices=['batch', 'sequential'],
                        default='batch',
                        help='Determines how to perform update in the SLAM algorithm.')
    parser.add_argument('-m',
                        '--movie-file',
                        type=str,
                        help='The full path to movie file to write the simulation animation to.',
                        default=None)
    parser.add_argument('--movie-fps',
                        type=float,
                        action='store',
                        help='The FPS rate of the movie to write.',
                        default=10.)
    parser.add_argument('--verbose',
                        type=bool,
                        help='Print intermediate information for Homework 3',
                        default=True)
    parser.add_argument('--plot2dcov', 
                        help='Plot 2d covariance')
    parser.add_argument('--iterative-solver', 
                        help='Solve per iteration')
    parser.add_argument('--no-iterative-solver',
                        help='Solve offline')  
    parser.add_argument('--solver', 
                        choices=['gn', 'lm'],
                        type=str, 
                        help='Solver type: GN or Levenberg-Marquard',
                        default=None)
    return parser.parse_args()

def validate_cli_args(args):
    if args.input_data_file and not os.path.exists(args.input_data_file):
        raise OSError('The input data file {} does not exist.'.format(args.input_data_file))

    if not args.input_data_file and not args.num_steps:
        raise RuntimeError('Neither `--input-data-file` nor `--num-steps` were present in the arguments.')


def main():
    args = get_cli_args()
    validate_cli_args(args)
    alphas = np.array(args.alphas) ** 2
    beta = np.array(args.beta)
    beta[1] = np.deg2rad(beta[1])
    Q = np.diag(beta**2)

    mean_prior = np.array([180., 50., 0.])
    Sigma_prior = 1e-12 * np.eye(3, 3)
    initial_state = Gaussian(mean_prior, Sigma_prior)

    if args.filter_name == 'sam':
        filter_model = Sam(initial_state, alphas, slam_type=args.filter_name,
                           data_association=args.data_association, update_type=args.update_type, Q=Q,
                           verbose=args.verbose)

    if args.input_data_file:
        data = load_data(args.input_data_file)
    elif args.num_steps:
        # Generate data, assuming `--num-steps` was present in the CL args.
        data = generate_input_data(initial_state.mu.T,
                                   args.num_steps,
                                   args.num_landmarks_per_side,
                                   args.max_obs_per_time_step,
                                   alphas,
                                   beta,
                                   args.dt)
    else:
        raise RuntimeError('')

    should_show_plots = True if args.animate else False
    should_write_movie = True if args.movie_file else False
    should_update_plots = True if should_show_plots or should_write_movie else False

    field_map = FieldMap(args.num_landmarks_per_side)

    fig = get_plots_figure('traj', should_show_plots, should_write_movie)

    movie_writer = get_movie_writer(should_write_movie, 'Simulation SLAM', args.movie_fps, args.plot_pause_len)
    progress_bar = FillingCirclesBar('Simulation Progress', max=data.num_steps)
    errors = []
    tt = []
    start_i = 1

    with movie_writer.saving(fig, args.movie_file, data.num_steps) if should_write_movie else get_dummy_context_mgr():
        for t in range(data.num_steps):
            # Used as means to include the t-th time-step while plotting.
            tp1 = t + 1

            # Control at the current step.
            u = data.filter.motion_commands[t]
            # Observation at the current step.
            z = data.filter.observations[t]

            # TODO SLAM predict(u)
            filter_model.predict(u)
            # TODO SLAM update
            filter_model.update(z)
            if args.iterative_solver:
                print(f'Solve per iter {t}')
                filter_model.solve()
            
            errors.append(filter_model.error())
            tt.append(t)
            
            progress_bar.next()
            if not should_update_plots:
                continue

            plt.cla()
            plot_field(field_map, z)
            plot_robot(data.debug.real_robot_path[t])
            plot_observations(data.debug.real_robot_path[t],
                              data.debug.noise_free_observations[t],
                              data.filter.observations[t])

            plt.plot(data.debug.real_robot_path[1:tp1, 0], data.debug.real_robot_path[1:tp1, 1], 'm')
            plt.plot(data.debug.noise_free_robot_path[1:tp1, 0], data.debug.noise_free_robot_path[1:tp1, 1], 'g')

            plt.plot([data.debug.real_robot_path[t, 0]], [data.debug.real_robot_path[t, 1]], '*r')
            plt.plot([data.debug.noise_free_robot_path[t, 0]], [data.debug.noise_free_robot_path[t, 1]], '*g')

            # TODO plot SLAM solution
            estimated_state = filter_model.graph.get_estimated_state()
            estimated_lndmark = np.array(estimated_state[start_i+1:])
            start_i = len(estimated_state)
            estimated_path = np.array([x for x in estimated_state if len(x) == 3]).reshape(-1, 3)
            last_state = estimated_path[-1]
            plt.plot(estimated_path[:, 0], estimated_path[:, 1], 'blue')
            #plt.scatter(estimated_lndmark[:, 0], estimated_lndmark[:, 1], c='blue', label='estimated landmarks')

            if should_show_plots:
                # Draw all the plots and pause to create an animation effect.
                plt.draw()
                plt.pause(args.plot_pause_len)

            if should_write_movie:
                movie_writer.grab_frame()

    progress_bar.finish()

    plt.show(block=True)

    fig = plt.figure()
    plt.plot(tt, errors)
    plt.title('CHI2 Error')
    plt.grid('on')
    fig.savefig('./errors.png')

    plt.spy(filter_model.adj_matrix())
    plt.savefig('./adjacency_matrix.png')
    plt.spy(filter_model.inform_matrix())
    plt.savefig('./information_matrix.png')
    if args.plot2dcov:
        plot2dcov(last_state[:-1], inv(filter_model.inform_matrix())[-2:, -2:].toarray(), color='purple', nSigma=3)    
    
    if not args.iterative_solver and args.solver == 'lm':
        print('LM offline solver')
        filter_model.graph.solve(mrob.LM)
        
    if not args.iterative_solver and args.solver == 'gn':
        print('GN offline solver')
        filter_model.graph.solve(mrob.GN)
        error = filter_model.graph.chi2()
        threshold = 1e-4
        i = 0
        while error > threshold:
            i += 1
            filter_model.graph.solve()
            error_next = filter_model.graph.chi2()
            if (error - error_next) > threshold:
                error = error_next
            else:
                break
        print(f'Error={error} number of steps = {i}')
        
if __name__ == '__main__':
    main()
