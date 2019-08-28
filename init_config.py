import os
import json


config = {
        	"localizer": {
        		"model": None,
        		"weights": None,
        		"threshold": 0.24
        	},
        	"cropper": {
        		"crop_percent": 0.25,
        		"force_square": True
        	},
        	"classifier": {
        		"model": None,
        		"weights": None,
        		"labels": None,
        		"threshold": 0.9,
        		"skip_classes": [11, 14, 17, 18, 19]
        	},
        	"detector": {
        		"images_path": None,
        		"sounds_path": None
        	}
        }


data_path = os.path.join(os.getcwd(), 'data/')
yolo_path = os.path.join(data_path, 'yolo/full/')
classifier_path = os.path.join(data_path, 'classifier/')

config['localizer']['model'] = os.path.join(yolo_path, 'trafficsigns.cfg')
config['localizer']['weights'] = os.path.join(yolo_path, 'trafficsigns.weights')

config['classifier']['model'] = os.path.join(classifier_path, 'trafficsigns.json')
config['classifier']['weights'] = os.path.join(classifier_path, 'trafficsigns.hdf5')
config['classifier']['labels'] = os.path.join(classifier_path, 'classes.txt')

config['detector']['images_path'] = os.path.join(classifier_path, 'classes/')
config['detector']['sounds_path'] = os.path.join(data_path, 'sounds/')


output_path = './cfg/config.json'

with open(output_path, 'w') as fp:
    json.dump(config, fp)
