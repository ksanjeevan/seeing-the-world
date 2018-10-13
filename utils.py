
import errno, os



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def setup_logging(logging_path='logs'):
    log_path = os.path.join(os.getcwd(), logging_path)
    mkdir_p(log_path)

    check_names = lambda y: y if y.isdigit() else -1
    get_ind = lambda x: int(check_names(x.split('_')[1]))

    run_counter = max(map(get_ind, os.listdir(log_path)), default=-1) + 1
    run_path = os.path.join(log_path, 'run_%s'%run_counter)

    mkdir_p(run_path)

    print('Logging ser up, to monitor training run:\n'
        '\t\'tensorboard --logdir=%s\'\n'%run_path)
    return run_path