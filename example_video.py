from detector.detector import create_detector_from_file

detector = create_detector_from_file('./cfg/example.config.json')
detector.detect_video_feed('./data/test/test.mp4', show_output=True, output='output.mp4')
