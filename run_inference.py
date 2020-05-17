"""
Method adapted from GMIC function `run_model` by
Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, and
Krzysztof J. Geras, which is licensed under a GNU Affero General Public License v3.0.
See: https://github.com/nyukat/GMIC/master/LICENSE
"""

import sys
import argparse

from gmic.modeling.run_model import start_experiment

print(sys.version, sys.platform, sys.executable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--model-path', default='models/')
    parser.add_argument('--data-path', default='sample_data/data.pkl')
    parser.add_argument('--image-path', default='sample_data/cropped_images/')
    parser.add_argument('--segmentation-path', default='sample_data/segmentation/')
    parser.add_argument('--output-path', default='sample_output/')
    parser.add_argument('--device-type', choices=['gpu', 'cpu'], default='cpu')
    parser.add_argument('--gpu-number', default=0, type=int)
    parser.add_argument('--model-index', choices=['1'], default='1')
    parser.add_argument('--visualization-flag', choices=[False, True], default=False)
    args = parser.parse_args()

    params = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path,
        "output_path": args.output_path,
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
    }

    start_experiment(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        model_index=args.model_index,
        parameters=params,
        turn_on_visualization=args.visualization_flag,
    )
