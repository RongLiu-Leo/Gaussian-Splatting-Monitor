from tqdm import tqdm

class ProgressBar:
    def __init__(self, total_iters, first_iter = 1):
        self.total_iters = total_iters
        self.bar = tqdm(range(first_iter, total_iters), desc="Training progress")

    def update(self,iteration,**kwargs):
        if iteration % 10 == 0:
            postfix = {key: f"{value:.5g}" if isinstance(value, (float, int)) else value for key, value in kwargs.items()}
            self.bar.set_postfix(postfix)
            self.bar.update(10)
        if iteration == self.total_iters:
            self.bar.close()
