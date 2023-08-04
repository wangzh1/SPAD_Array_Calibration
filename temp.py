import argparse

parser = argparse.ArgumentParser(description='aaa')
parser.add_argument('type', type=str, help='calibration or inference')
args = parser.parse_args()
if args.type == 'infer':
    print('hello')
elif args.type == 'cali':
    print('hi')
else:
    print('noinput')