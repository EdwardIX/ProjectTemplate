import streamlit as st
from comm import SocketWeb

# 初始化 SocketWeb
socket_web = SocketWeb()

# Streamlit 页面展示
st.title("实验监控系统")

if st.button('刷新数据'):
    st.rerun()  # 刷新页面

# 显示实验列表
st.subheader("所有实验")
experiment_list = socket_web.get_experiment_list()

experiment_list.sort(key=(lambda e: (e['finished'], e['start_time']))) # Put finished exps on the back
for experiment in experiment_list:
    with st.expander(f"实验: {experiment['name']} {'(已结束)' if experiment['finished'] else ''}"):
        st.markdown(f"**开始时间:** {experiment['start_time']}")
        st.markdown(f"**任务数量:** {experiment['num_tasks']}")

        # 获取实验详细信息
        experiment_status = socket_web.get_experiment_status(experiment['id'])
        
        st.write("任务详情:")
        status_table = {}
        for j in range(experiment_status['max_repeat']):
            status_table[j] = [experiment_status['task_status'].get((i, j), "--") for i in range(experiment['num_tasks'])]
        st.table(status_table)

        # 停止实验按钮
        if not experiment['finished']:
            if st.button(f"停止实验 {experiment['id']}", key=f"stop_exp_{experiment['id']}"):
                suc, msg = socket_web.cmd_stop_experiment(experiment['id'])
                if suc:
                    st.success(f"实验 {experiment['id']} 已停止")
                else:
                    st.error(f"实验 {experiment['id']} 停止失败：{msg}")

# 显示正在运行的任务列表
st.subheader("正在运行的任务")
task_list = socket_web.get_task_list()

for task in task_list:
    with st.expander(f"任务 {task['exp_name']} : {task['index']}"):
        st.markdown(f"**所属实验名称:** {task['exp_name']}")
        st.markdown(f"**所属实验开始时间:** {task['run_time']}")
        st.markdown(f"**任务运行节点:**")
        for k, v in task['gpuinfo'].items():
            st.markdown(f"* {k}: {','.join(map(str, v))}")

        # 停止任务按钮
        if st.button(f"停止任务 {task['id']}", key=f"stop_task_{task['id']}"):
            suc, msg = socket_web.cmd_stop_task(task['id'])
            if suc:
                st.success(f"任务 {task['id']} 已停止")
            else:
                st.error(f"任务 {task['id']} 停止失败：{msg}")

# 显示 Runner 状态
st.subheader("Runner 状态")
runner_status = socket_web.get_runner_status()

for runner in runner_status:
    with st.expander(f"Runner: {runner['id']}"):
        st.code(runner['gpustat'])