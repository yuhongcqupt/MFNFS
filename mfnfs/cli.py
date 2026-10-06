from .experiment import build_arg_parser, run_experiment

def main():
    args = build_arg_parser().parse_args()
    run_experiment(args)
if __name__ == '__main__':
    main()
