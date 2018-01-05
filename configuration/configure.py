import json
from colorama import init, deinit
from colorama import Fore, Back, Style


def config_file_to_detection_pipeline(config_path):
    with open(config_path, 'r') as fp:
        config_json = json.load(fp)

    localizer_config = config_json['localizer']
    cropper_config = config_json['cropper']
    classifier_config = config_json['classifier']

def create_config_file():
    # Configuration template
    configuration = {
        'localizer': {
            'model': None,
            'weights': None,
            'threshold': None,
        },
        'cropper': {
            'crop_percent': None,
            'force_square': None,
        },
        'classifier': {
            'model': None,
            'weights': None,
            'labels': None,
            'threshold': None,
        }
    }

    init()

    print(Fore.GREEN + Style.BRIGHT)
    print(' Object Detector Configuration')
    print('⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻\n' + Style.RESET_ALL)

    for key in configuration.keys():
        dict_entry = configuration[key]
        for subkey in dict_entry.keys():
            print(Style.BRIGHT + Fore.CYAN)
            cfg_value = input('[' + key + '/' + subkey + ']: ' + Style.RESET_ALL)
            configuration[key][subkey] =  cfg_value

    print('\n\n')
    output_path = input(Fore.YELLOW + Style.BRIGHT + 'Enter the output path for the configuration file: ' + Style.RESET_ALL)

    with open(output_path, 'w') as fp:
        json.dump(configuration, fp)

    deinit()


if __name__ == '__main__':
    config_file_to_detection_pipeline('config.json')
    #create_config_file()
