
import subprocess

class BaseModule():
    def __init__(self, cfg, logger):
        self.logger = logger
        if isinstance(cfg, list):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            for key, value in cfg.items():
                if not hasattr(self.__class__, key) or not isinstance(getattr(self.__class__, key), property):
                    setattr(self, key, value)
        

    def run_command_with_realtime_output(self, cmd):
        """
        Run the specified command and output the results in real-time.

        :param cmd: The command string to run.
        :return: The exit code of the command.
        """
        self.logger.info(f"Running command: {cmd}")
        # Start the process
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                self.logger.verbose(output.strip())  

        # Read any remaining error output
        stderr_output = process.stderr.read()
        if stderr_output:
            self.logger.error("Error Output:")  
            self.logger.error(stderr_output.strip())

        # Return the exit code
        return process.returncode
