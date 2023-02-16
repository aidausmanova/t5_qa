"""
This script offers the CLI interface for the entire project.
"""

from pathlib import Path
import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def get_package_root():
    """
    This function fetches the path to the parent directory and returns the path as a string.

    Returns:
        path ():    Path to the root directory of the project as a string.
    """

    return str(Path(__file__).parent.resolve().parent.resolve().parent.resolve())


def initialise_environment(global_properties, simulator_properties):
    """
    This function initialises the AI2Thor environment to be used throughout the project.

    :param global_properties:       It contains the global properties for the entire project.
    :type global_properties:        dict
    :param simulator_properties:    It contains the properties specific to the simulator.
    :type simulator_properties:     dict
    :return environment:            This is the object of the environment containing the agent configuration among other information.
    :rtype:                         Environment
    """

    environment = Environment(global_properties, simulator_properties)
    return environment


def _init(**kwargs):
    """
    Initialise the global system properties.
    """

    # Initializing the parent root directory for path configuration.
    kwargs['package_root'] = get_package_root()

    # Initializing the Global Variables which will be available throughout the project.
    global_variables = GlobalVariables(**kwargs)
    global_variables.load_configuration_properties()

    return global_variables


# Command line interface (CLI) main
@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli1(**kwargs):
    _init(**kwargs)

    click.echo("\nEntry point '1' for Command Line Interface. Type 'project --help' for details.\n")


@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli2(**kwargs):
    _init(**kwargs)

    click.echo("\nEntry point '2' for Command Line Interface. Type 'project --help' for details.\n")


@click.group(chain=True, help='Command line tool for pyetl.', invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
def cli3(**kwargs):
    _init(**kwargs)

    click.echo("\nEntry point '3' for Command Line Interface. Type 'project --help' for details.\n")


@cli1.command('training', context_settings=CONTEXT_SETTINGS)
@click.option('-s', '--start', is_flag=True, help='Start training')
@click.option('-d', '--dataset', help='Start training')
@click.option('-e', '--epochs', help='Choose specific learning rate')
# @click.option('-g', '--gpu', help='Give GPU options as 0 or 1')
# @click.option('-c', '--checkpoint', help='Training on top of a checkpoint')
def start_local_training(**kwargs):
    _init(**kwargs)

    """Start training on local machine"""
    print("Start training")

    if kwargs['dataset'] == 'conceptnet':
        print("Dataset = " + kwargs['dataset'])
    elif kwargs['dataset'] == 'tellmewhy':
         print("Dataset = " + kwargs['dataset'])
    
    if kwargs['epochs']:
        global_properties['epochs'] = int(kwargs['epochs'])
    else:
        global_properties['epochs'] = 3

    # if kwargs['gpu']:
    #     print("GPU = " + kwargs['gpu'])
    #     global_properties['gpu'] = kwargs['gpu']
    # else:
    #     print("GPU = " + "1")
    #     global_properties['gpu'] = "1"

    # if kwargs['checkpoint']:
    #     global_properties['checkpoint'] = kwargs['checkpoint']
    # else:
    #     global_properties['checkpoint'] = None

    

    # environment = initialise_environment(global_properties, simulator_properties)
    # print('Simulator Environment Initialized')

    # if kwargs['start']:
    #     print('Staring environment')
    #     environment.start()


cli = click.CommandCollection(sources=[cli1, cli3])

if __name__ == '__main__':
    cli()
