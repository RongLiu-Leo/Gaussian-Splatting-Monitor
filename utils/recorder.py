import torch
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model.base import BaseModule

class Recorder(BaseModule):
    def __init__(self, cfg, logger, info_path, first_iter, max_iter):
        super().__init__(cfg, logger)
        self.info_path = info_path
        self.data = {}
        self.save_iterations = {}
        self.value_type = {}
        self.record_path = {}
        for record_item in self.cfg:
            name = record_item.get("name")
            self.record_path[name] = info_path / name
            self.record_path[name].mkdir(parents=True, exist_ok=True)
            self.data[name] = []
            
            # value type
            value_type = record_item.get("value_type")
            self.value_type[name] = value_type
            
            # save intervals
            if "save_iterations" not in record_item:
                save_intervals = record_item.get("save_intervals")
                record_item["save_iterations"] = list(range(first_iter, max_iter + 1, save_intervals))
            self.save_iterations[name] = record_item.get("save_iterations")
    
    def snapshot(self, name, value):
        self.data[name].append(value)

    def update(self, iteration):
        for name in self.data.keys():
            if self.should_process(iteration, name):
                if self.value_type[name] == "number":
                    store_func = self.store_number
                elif self.value_type[name] == "1darray":
                    store_func = self.store_1darray
                elif self.value_type[name] == "imageDict":
                    store_func = self.store_imageDict
                elif self.value_type[name] == "tensor":
                    store_func = self.store_tensor
                store_func(name = name, data = self.data[name], iteration = iteration, save_folder = self.record_path[name])               
                self.data[name] = [] # clear temporary data

    def should_process(self, iteration, name):
        return iteration in self.save_iterations[name]
    
    @staticmethod
    def store_number(name, data, iteration, save_folder):
        # save values
        save_file = save_folder / "value.txt"
        with open(str(save_file), "a") as file:
            for d in data:
                file.write(str(d) + "\n")

        # plot line chart
        save_chart = Path(save_folder) / "line_chart.png"
        values = []
        with open(save_file, 'r') as file:
            for line in file:
                try:
                    value = float(line.strip())
                    values.append(value)
                except ValueError:
                    print(f"Skipping invalid line: {line}")

        plt.plot(values)
        plt.xlabel('Iteration')
        plt.ylabel(name)
        plt.title(f'Line Chart of {name}')
        plt.savefig(str(save_chart))
        plt.close()
    
    @staticmethod
    def store_1darray(name, data, iteration, save_folder):
        plt.figure()
        plt.hist(data, bins=100)
        plt.yscale('log')
        plt.title(f'Distribution Chart of {name}')
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.savefig(str(Path(save_folder) / f"distribution_chart_{iteration}.png"))
        plt.close()

    @staticmethod
    def store_imageDict(name, data, iteration, save_folder):
        for d in data:
            for filename, value in d.items():
                image_pil = transforms.ToPILImage()(value.squeeze(0))
                image_pil.save(str(Path(save_folder) / f"{filename}_{iteration}.png"))
    
    @staticmethod
    def store_tensor(name, data, iteration, save_folder):
        save_path = Path(save_folder) / f"{name}_{iteration}.pt"
        torch.save(data, save_path)

