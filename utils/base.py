
import subprocess
import threading

class BaseModule():
    def __init__(self, cfg, logger):
        self.logger = logger
        if isinstance(cfg, list):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            for key, value in cfg.items():
                if not hasattr(self.__class__, key) or not isinstance(getattr(self.__class__, key), property):
                    setattr(self, key, value)
        

    def _read_stream(self, stream, logger_method):
        """
        Read from a stream and log using the specified logger method.

        :param stream: The stream to read from.
        :param logger_method: The logger method to use for logging.
        """
        for line in iter(stream.readline, ''):
            logger_method(line.strip())

    def run_command_with_realtime_output(self, cmd):
        """
        Run the specified command and output the results in real-time.

        :param cmd: The command string to run.
        :return: The exit code of the command.
        """
        self.logger.info(f"Running command: {cmd}")
        # Start the process
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Create and start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=self._read_stream, args=(process.stdout, self.logger.verbose))
        stderr_thread = threading.Thread(target=self._read_stream, args=(process.stderr, self.logger.error))
        stdout_thread.start()
        stderr_thread.start()

        # Wait for threads to finish
        stdout_thread.join()
        stderr_thread.join()

        # Wait for the process to finish and get the exit code
        exit_code = process.wait()

        # Return the exit code
        return exit_code
