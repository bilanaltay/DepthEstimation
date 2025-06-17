import argparse

from predictor import DepthEstimationModel


def main():
    parser =argparse.ArgumentParser(description="Depth estimation using ZoeD_N")
    parser.add_argument("-i","--image",help="Image path")
    parser.add_argument("-o","--output",help="Output path")
    args = parser.parse_args()

    model = DepthEstimationModel()
    result = model.calculate_depthmap(args.image, args.output)
    print(result)

if __name__ == "__main__":
    main()