import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank", default=-1, type=str, help="node rank for distributed training"
    )

    return parser.parse_args()


if __name__ == "__main__":
    _args = get_args()
    print(_args.local_rank, type(_args.local_rank))
