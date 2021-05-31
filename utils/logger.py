from tensorboardX import SummaryWriter
import logging

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def logs_training(self, reduced_losses, grad_norms, learning_rates, durations, iteration):
        for i in range(len(reduced_losses)):
            self.log_training(reduced_losses[i], grad_norms[i], learning_rates[i], durations[i],iteration+i)

    def logs_validation(self, reduced_losses, iterations):
        for i in range(1, len(reduced_losses)):
            self.log_validation(reduced_losses[i], iterations[i])

    def log_training(self, total_loss, accuracy, iteration):
        self.add_scalar("training.loss", total_loss, iteration)
        self.add_scalar("training.accuracy", accuracy, iteration)

    def log_validation(self, total_loss, accuracy, test_loss, test_accuracy, iteration):
        self.add_scalar("validation.loss", total_loss, iteration)
        self.add_scalar("validation.accuracy", accuracy, iteration)
        self.add_scalar("test_on_target.loss", test_loss, iteration)
        self.add_scalar("test_on_target.accuracy", test_accuracy, iteration)


def create_and_configer_logger(log_name='log_file.log', level=logging.INFO):
    """
    Sets up a logger that works across files.
    The logger prints to console, and to log_name log file.

    Example usage:
        In main function:
            logger = create_and_configer_logger(log_name='myLog.log')
        Then in all other files:
            logger = logging.getLogger(_name_)

        To add records to log:
            logger.debug(f"New Log Message. Value of x is {x}")

    Args:
        log_name: str, log file name

    Returns: logger
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=level,
        format='\n' + '[%(asctime)s - %(levelname)s] {%(pathname)s:%(lineno)d} -' + '\n' + ' %(message)s' + '\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level=level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger