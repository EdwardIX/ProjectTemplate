import streamlit as st

# 模拟 SocketWeb 类
class SocketWeb:
    def __init__(self):
        pass

    def get_task_list(self):
        # 返回所有正在运行的任务
        return [{"id": 1, "status": "running"}, {"id": 2, "status": "waiting"}]

    def get_experiment_list(self):
        # 返回所有实验
        return [{"id": 1, "name": "Experiment 1", "start_time": "2024-10-25", "num_tasks": 3}]

    def get_task_status(self, identifier):
        # 返回指定任务的状态
        return {"id": identifier, "status": "running"}

    def get_experiment_status(self, identifier):
        # 返回指定实验的状态
        return {"id": identifier, "name": "Experiment 1", "start_time": "2024-10-25", "num_tasks": 3, "tasks": [{"id": 1, "status": "running"}]}

    def get_runner_status(self):
        # 返回 Runner 状态
        return [{"name": "Runner 1", "total_nodes": 4, "busy_nodes": 2, "nodes": [{"id": 1, "utilization": 70, "memory_left": 1024}]}]

    def cmd_stop_task(self, identifier):
        print(f"Stopping task {identifier}")

    def cmd_stop_experiment(self, identifier):
        print(f"Stopping experiment {identifier}")

# 初始化 SocketWeb
socket_web = SocketWeb()

# Streamlit 页面展示
st.title("实验监控系统")

# 显示实验列表
st.subheader("正在运行的实验")
experiment_list = socket_web.get_experiment_list()

for experiment in experiment_list:
    with st.expander(f"实验: {experiment['name']}"):
        st.write(f"开始时间: {experiment['start_time']}")
        st.write(f"任务数量: {experiment['num_tasks']}")

        # 获取实验详细信息
        experiment_details = socket_web.get_experiment_status(experiment['id'])
        st.write("任务详情:")
        for task in experiment_details["tasks"]:
            st.write(f"任务 {task['id']} - 状态: {task['status']}")

        # 停止实验按钮
        if st.button(f"停止实验 {experiment['name']}", key=f"stop_exp_{experiment['id']}"):
            socket_web.cmd_stop_experiment(experiment['id'])
            st.success(f"实验 {experiment['name']} 已停止")

# 显示正在运行的任务列表
st.subheader("正在运行的任务")
task_list = socket_web.get_task_list()

for task in task_list:
    with st.expander(f"任务 {task['id']}"):
        task_status = socket_web.get_task_status(task['id'])
        st.write(f"状态: {task_status['status']}")
        
        # 停止任务按钮
        if st.button(f"停止任务 {task['id']}", key=f"stop_task_{task['id']}"):
            socket_web.cmd_stop_task(task['id'])
            st.success(f"任务 {task['id']} 已停止")

# 显示 Runner 状态
st.subheader("Runner 状态")
runner_status = socket_web.get_runner_status()

for runner in runner_status:
    with st.expander(f"Runner: {runner['name']}"):
        st.write(f"GPU 节点总数: {runner['total_nodes']}")
        st.write(f"繁忙节点: {runner['busy_nodes']}")
        
        for node in runner["nodes"]:
            st.write(f"GPU {node['id']} - 利用率: {node['utilization']}%, 剩余内存: {node['memory_left']} MB")

