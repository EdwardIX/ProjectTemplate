from task import Task

class Experiment:
    def __init__(self, config_list, exp_name="Test"):
        """
        Create a Group of Task
        exp_name: the name of this experiment
        """

        self.exp_name = exp_name
        self.run_time = time.strftime('%y.%m.%d-%H.%M.%S')
        self.tasks:List[Task] = []     # List of tasks

        os.makedirs(os.path.join("runs", self.exp_name, self.run_time), exist_ok=True)
        
        for config in config_list:
            self.tasks.append(Task(config['config'], config['runreq']))
        
        self.save_task()
    
    def load_task(self):
        print(f"\033[33m############  Warning: Load Task List from existing file. Current Task List replaced  ############\033[0m")
        with open(os.path.join("runs", self.exp_name, ))

    def save_task(self):
        with open(os.path.join("runs", self.exp_name, "tasks.json"), "w") as f:
            json.dump([
                t.args for t in self.tasks
            ], f, indent=2)
        with open(os.path.join("runs", self.exp_name, self.run_time, "runinfo.json"), "w") as f:
            json.dump({
                "Command": "python " + " ".join(sys.argv[1:]),

                "TaskRequirement": [t.reqs for t in self.tasks],
            }, f, indent=2)

    def add_task(self, config):
        self.tasks.append(Task(config['config'], config['runreq']))