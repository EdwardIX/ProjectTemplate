import os
import sys
import time
import types
import pickle
import logging
import inspect
import traceback
import pandas as pd
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .plot import plot_heatmap, plot_lines

def colored(text, color):
    # ANSI color codes for text
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'reset': '\033[0m',
    }
    return f"{colors[color]}{text}{colors['reset']}"

class ColoredFormatter(logging.Formatter):
    
    def format(self, record):
        log_message = super(ColoredFormatter, self).format(record)
        if record.levelno == logging.INFO:
            return colored(f"[I] {log_message}", 'green')
        elif record.levelno == logging.WARNING:
            return colored(f"[W] {log_message}", 'yellow')
        elif record.levelno == logging.ERROR:
            return colored(f"[E] {log_message}", 'red')
        else:
            return log_message

def to_numpy(x):
    if isinstance(x, dict):
        return {k:to_numpy(v) for k,v in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def format_list(l):
    str_l = []
    for x in l:
        if isinstance(x, (int, float)):
            str_l.append(f"{x:10.3e}")
        else:
            str_l.append(f"{x.item():10.3e}")
    return '[' + ' '.join(str_l) + ']'

def _recursive_debug(data, indent=0, max_display=10, super_is_dict=False):
    indent_str = '  ' * indent if not super_is_dict else ''
    if isinstance(data, list) or isinstance(data, types.GeneratorType):
        if isinstance(data, types.GeneratorType):
            data = list(data)
            print(f"{indent_str}{colored('Generator', 'red')} (length: {len(data)}):", file=sys.stderr)
        else:
            print(f"{indent_str}{colored('List', 'cyan')} (length: {len(data)}):", file=sys.stderr)
        if len(data) <= max_display * 2:
            for item in data:
                _recursive_debug(item, indent + 1)
        else:
            for item in data[:max_display]:
                _recursive_debug(item, indent + 1)
            _recursive_debug(f"...: (omitted {len(data) - max_display * 2} items)", indent + 1)
            for item in data[-max_display:]:
                _recursive_debug(item, indent + 1)
    
    elif isinstance(data, tuple):
        has_newline = any(isinstance(item, (list, tuple, dict, np.ndarray, torch.Tensor)) for item in data)
        if has_newline:
            print(f"{indent_str}{colored('Tuple', 'blue')} (length: {len(data)}):", file=sys.stderr)
            if len(data) <= max_display * 2:
                for item in data:
                    _recursive_debug(item, indent + 1)
            else:
                for item in data[:max_display]:
                    _recursive_debug(item, indent + 1)
                _recursive_debug(f"...: (omitted {len(data) - max_display * 2} items)", indent + 1)
                for item in data[-max_display:]:
                    _recursive_debug(item, indent + 1)
        else:
            print(f"{indent_str}{colored('Tuple', 'blue')}: {data}", file=sys.stderr)
    
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        if isinstance(data, np.ndarray):
            data_type = 'ndarray'
        elif isinstance(data, torch.Tensor):
            data_type = 'Tensor'
        print(f"{indent_str}{colored(data_type, 'magenta')} (shape: {data.shape}, mean: {data.mean().item():.6f}, min: {data.min().item():.6f}, max: {data.max().item():.6f})", file=sys.stderr)
    
    elif isinstance(data, dict):
        print(f"{indent_str}{colored('Dict', 'yellow')} (length: {len(data)}):", file=sys.stderr)
        for key, value in data.items():
            print(f"{indent_str}  {colored(key, 'green')}:", file=sys.stderr, end='  ')
            _recursive_debug(value, indent + 2, super_is_dict=True)
    
    else:
        print(f"{indent_str}{data}", file=sys.stderr)

# Tensorboard：自动定制输出信息，保存信息
class Logger:
    def __init__(self):
        self.taskstr = None
        self.writer = None
        self.logger = None
        self.logdir = None
        self.timers = {}
        self.history = defaultdict(dict)
        self.logged_messages = set()
        self.log_images = True
    
    def initialize(self, log_dir, taskid, repeatid):
        self.taskstr = f" [{taskid} - {repeatid}] "
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logdir = log_dir

        formatter = ColoredFormatter('%(message)s')
        # formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 添加一个控制台处理器
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _get_caller_info(self):
        # 获取调用者的帧对象
        caller_frame = inspect.stack()[2]
        caller_file = caller_frame[1]
        caller_function = caller_frame[3]
        caller_line = caller_frame[2]
        return f"{caller_file}:{caller_line}"

    def info(self, *messages, once=False):
        if self.logger:
            caller_info = self._get_caller_info()
            if once:
                if caller_info in self.logged_messages:
                    return
                self.logged_messages.add(caller_info)
            message = " ".join(map(str, messages))
            self.logger.info(f"[{caller_info}] {message}")

    def warning(self, *messages, once=False):
        if self.logger:
            caller_info = self._get_caller_info()
            if once:
                if caller_info in self.logged_messages:
                    return
                self.logged_messages.add(caller_info)
            message = " ".join(map(str, messages))
            self.logger.warning(f"[{caller_info}] {message}")

    def error(self, *messages):
        if self.logger:
            caller_info = self._get_caller_info()
            message = " ".join(map(str, messages))
            for line in traceback.format_stack()[:-1]:
                print(line.strip(), file=sys.stderr)
            self.logger.error(f"[{caller_info}] {message}")
            exit(1)
    
    def debug(self, *items, once=False):
        if self.logger:
            caller_info = self._get_caller_info()
            if once:
                if caller_info in self.logged_messages:
                    return
                self.logged_messages.add(caller_info)
            self.logger.info(f"[{caller_info}] debug with {len(items)} items:")
            for item in items:
                _recursive_debug(item)
    
    def add_scalars(self, scalars, global_step=None, group=None, verbose=True):
        if not self.writer:
            return
        for name, value in scalars.items():
            try:
                if value is None:
                    value = np.nan
                scalars[name] = float(value)
            except Exception:
                self.error("Cannot Convert scalar to float: recheck values in logger.add_scalars")
            
            if group is not None:
                name = f"{group}/{name}"
            self.writer.add_scalar(name, value, global_step=global_step)
            self.history[global_step][name] = float(value)
        
        if not isinstance(verbose, bool):
            assert isinstance(verbose, int) and verbose > 0
            verbose = (global_step % verbose == 0) if global_step is not None else False
        
        if verbose: 
            names = "|"
            vals = "|"

            if global_step is not None:
                names += f" step |"
                vals += f"{global_step:6d}|"
            
            for name, value in scalars.items():
                if len(name) < 10:
                    res = 10 - len(name)
                    names += " " * (res - (res // 2)) + name + " " * (res // 2) + "|"
                    vals += f'{value:10.3e}|'
                else:
                    res = len(name) - 10
                    names += f"{name}|"
                    vals +=  " " * (res - (res // 2)) + f'{value:10.3e}' + " " * (res // 2) + "|"
            
            length = len(names)
            res = length - len(self.taskstr)
            print("-" * (res - (res // 2)) + self.taskstr + "-" * (res // 2))
            print(names)
            print("-" * length)
            print(vals)
            print("-" * length)
            print("")

    def save_figure_data(self, figtype, name, data, global_step=None, group=None):
        savepath = os.path.join(self.logdir, 'figures')
        if group is not None:
            savepath = os.path.join(savepath, group)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        savepath = os.path.join(savepath, f"{figtype}-{name.replace('/', '-')}-step{0 if global_step is None else global_step}.pkl")
        
        with open(savepath, 'wb') as f:
            pickle.dump(data, f)

    def add_image(self, name, image, global_step=None, dataformats='CHW'):
        if self.writer and self.log_images:
            self.writer.add_image(name, image, global_step=global_step, dataformats=dataformats)
    
    def add_heatmap(self, name, x, y, z, global_step=None, group=None, **kwargs):
        if self.writer and self.log_images:
            data = to_numpy(x), to_numpy(y), to_numpy(z)
            fig = plot_heatmap(*data, **kwargs)
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            self.save_figure_data('heatmap', name, data, global_step, group)
            if group is not None:
                name = f"{group}/{name}"
            self.add_image(name, img, global_step, dataformats='HWC')
            
    
    def add_lines(self, name, x, y, global_step=None, group=None, **kwargs):
        if self.writer and self.log_images:
            data = to_numpy(x), to_numpy(y)
            fig = plot_lines(*data, **kwargs)
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            self.save_figure_data('lines', name, data, global_step, group)
            if group is not None:
                name = f"{group}/{name}"
            self.add_image(name, img, global_step, dataformats='HWC')

    # def add_line(self, name, value, index=None):
    #     if self.writer:
    #         if index is None:
    #             index = np.arange(len(value))
    #         for i in range(len(index)):
    #             self.writer.add_scalar(name, value[i], index[i])
    
    def timer_start(self, name="default", reset=True, verbose=False):
        self.timers[name] = (0 if reset else self.timers.get(name, 0)) - time.time()
        if verbose:
            caller_info = self._get_caller_info()
            namestr = name if isinstance(verbose, bool) else verbose
            print(colored(f"[TIMER] [{caller_info}] start timing {namestr}", 'blue'))

    def timer_stop(self, name="default", verbose=False):
        self.timers[name] = self.timers[name] + time.time()
        if verbose:
            caller_info = self._get_caller_info()
            namestr = name if isinstance(verbose, bool) else verbose
            print(colored(f"[TIMER] [{caller_info}] {namestr} took {self.timers[name]} s", 'blue'))
        return self.timers[name]
    
    def timer_get(self, name="default", verbose=False):
        # Add current time If The current timer is running
        timer_time = self.timers[name] + time.time() if self.timers[name] < 0 else self.timers[name]
        if verbose:
            caller_info = self._get_caller_info()
            namestr = name if isinstance(verbose, bool) else verbose
            print(colored(f"[TIMER] [{caller_info}] {namestr} took {timer_time} s", 'blue'))
        return timer_time

    def timer_reset(self, name="default"):
        self.timers[name] = 0
    
    def summary(self, path):
        if len(self.history) == 0:
            return {}

        # 获取所有变量名并按名称排序
        variables = set()
        for step_data in self.history.values():
            variables.update(step_data.keys())
        variables = sorted(list(variables))

        # 构建DataFrame
        rows = []
        for step, step_data in self.history.items():
            row = {'step': step}
            row.update({var: step_data.get(var, np.nan) for var in variables})
            rows.append(row)
        df = pd.DataFrame(rows)

        # 按照step排序
        df = df.sort_values(by='step')

        # 计算统计信息
        statistics = {}
        for var in variables:
            dropna_values = df[var].dropna()
            if not dropna_values.empty:
                statistics[var] = {'last': dropna_values.iloc[-1],
                                   'last_5': dropna_values.tail(5).mean(),
                                   'last_10': dropna_values.tail(10).mean(),
                                   'all': dropna_values.mean()}
            else:
                statistics[var] = {'last': np.nan,
                                   'last_5': np.nan,
                                   'last_10': np.nan,
                                   'all': np.nan}

        # 导出为逗号分隔符文件
        df.to_csv(path, index=False)

        return statistics

logger = Logger()