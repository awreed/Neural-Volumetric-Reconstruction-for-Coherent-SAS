from utils import save_3d_matplotlib_scene
import argparse
import numpy as np

def config_parser():
    parser = argparse.ArgumentParser(description="Sample the network at arbitrary resolution")
    parser.add_argument('--scene_npy_file', required=True,
                        help='scene npy file to render')
    parser.add_argument('--output_dir', required=True,
                        help='output directory')
    parser.add_argument('--output_name', required=True,
                        help='file name')
    parser.add_argument('--thresh', required=False, type=float, default=2,
                        help='filter out points < mean + thresh*std')
    parser.add_argument('--downsample_factor', required=False, type=int, default=None,
                        help='Downsample the scene prior to plotting (helps with matplotlib performance issues')
    parser.add_argument('--elev', required=False, type=int, default=2,
                        help="Viewing elevation")
    parser.add_argument('--num_angles', required=False, type=int, default=4,
                        help='Number of uniform angles between 0-360 degrees to plot')

    return parser

if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    scene = np.load(args.scene_npy_file)
    print("Loaded scene with shape", scene.shape)

    print("Rendering 3D scene using matplotlib...")
    save_3d_matplotlib_scene(scene, args.output_dir, args.output_name, elev=args.elev, num_angles=args.num_angles,
                             thresh=args.thresh, downsample_factor=args.downsample_factor)
    print("Done.")