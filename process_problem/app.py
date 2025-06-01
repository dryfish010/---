from flask import Flask, request, jsonify, render_template
import random
import matplotlib.pyplot as plt
import matplotlib
import time
import base64
from io import BytesIO
from ga import run_ga  # 確保你有 `ga.py` 模組
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
from collections import defaultdict




app = Flask(__name__, static_folder="static")


@app.route('/')
def index():
    return render_template('form.html')

@app.route('/run_ga', methods=['POST'])
def process_jobs():
    data = request.get_json()
    
    job_inputs = data.get("jobs", {})
    machine_times = data.get("machine_times", {})  # ✅ 確保接收 machine_times
    machine_capacity=data.get("machine_capacity",{})
    #print(machine_capacity)
    #print(machine_times)
    if not job_inputs:
        return jsonify({"error": "No job data provided"}), 400

    # 轉換數據格式
    jobs = {int(job_id): job_info["machines"] for job_id, job_info in job_inputs.items()}
    job_deadlines = {int(job_id): int(job_info["days"]) for job_id, job_info in job_inputs.items()}

    #print("收到的 machine_times:", machine_times)  # ✅ 確保後端有接收機器時間

    # 🔹 確保 `run_ga()` 接收 `machine_times`
    result = run_ga(jobs, job_deadlines, machine_times,machine_capacity)
    return jsonify(result)


def extract_all_machines(jobs):
    """ 從所有工作提取機器名稱清單 """
    machine_set = set()
    for steps in jobs.values():
        machine_set.update(steps)
    return list(machine_set)


def initialPop(machine_list, POP_SIZE):
    """ 生成初始族群 """
    return [random.sample(machine_list, len(machine_list)) for _ in range(POP_SIZE)]


def efficiency(schedule, jobs, job_deadlines, machine_times, machine_capacity):
    #計算適應度：考慮步數、機器處理時間、遲交與機器最大承受數量 
    total_steps, total_late_penalty, total_time, machine_overload_penalty = 0, 0, 0, 0
    machine_usage = {m: 0 for m in machine_capacity}  # 記錄機器的負載
    
    
    for job_id, machine_list in jobs.items():
        sorted_steps = sorted(machine_list, key=lambda m: schedule.index(m))
        steps, completion_time = 0, 0

        for i in range(1, len(sorted_steps)):
            
            steps += abs(schedule.index(sorted_steps[i]) - schedule.index(sorted_steps[i - 1]))
            completion_time += machine_times[sorted_steps[i]]

            # 記錄機器使用次數
            machine_usage[sorted_steps[i]] += 1
        
        total_steps += steps
        total_time += completion_time

        # 超過可處理天數則加懲罰
        late_days = max(0, completion_time / 24 - job_deadlines[job_id])
        total_late_penalty += late_days * 5  # 懲罰權重可調整

    # 檢查機器是否過載
    for machine, usage in machine_usage.items():
        if usage > machine_capacity[machine]:  # 超過可處理數量
            machine_overload_penalty += (usage - machine_capacity[machine]) * 10  # 懲罰可調整

    # 目標：平衡步數、處理時間、遲交懲罰、機器負載
    return 1 / (1 + total_steps + total_time / 10 + total_late_penalty + machine_overload_penalty)

def selectGA(population, jobs, job_deadlines,machine_times, machine_capacity):
    """ 父代選擇（竟賽選擇） """
    return max(random.sample(population, 5), key=lambda ind: efficiency(ind, jobs, job_deadlines, machine_times, machine_capacity))


def crossover(parent1, parent2):
    """ 交叉操作 """
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))

    child = [-1] * size
    child[p1:p2] = parent1[p1:p2]

    remaining = [gene for gene in parent2 if gene not in child]
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining.pop(0)

    return child


def mutate(offspring, MUTATION_RATE):
    """ 突變操作 """
    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(len(offspring)), 2))
        sub = offspring[i:j]
        random.shuffle(sub)
        offspring[i:j] = sub

    return offspring


def nextGA(parents, pop_size, jobs, MUTATION_RATE):
    """ 產生子代 """
    offsprings = []
    while len(offsprings) < pop_size:
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child, MUTATION_RATE)
        offsprings.append(child)
    return offsprings


def GA(jobs, job_deadlines, POP_SIZE, GENS, MUTATION_RATE, machine_list, machine_times, machine_capacity):
    population = initialPop(machine_list, POP_SIZE)
    best_score, best_solution, total_history = float('-inf'), None, []
    best_step_sequence = {}  # ✅ 儲存工作步驟

    for gen in range(GENS):
        scores = [efficiency(ind, jobs, job_deadlines, machine_times, machine_capacity) for ind in population]
        best_idx = scores.index(max(scores))
        total_history.append(1 / scores[best_idx])

        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_solution = population[best_idx]

            # ✅ 記錄最佳解的工作步驟
            for job_id, machines in jobs.items():
                sorted_machines = sorted(machines, key=lambda m: best_solution.index(m))
                best_step_sequence[job_id] = [
                    {"step": i + 1, "machine": machine, "processing_time": machine_times.get(machine, 0)}
                    for i, machine in enumerate(sorted_machines)
                ]

        parents = [selectGA(population, jobs, job_deadlines, machine_times, machine_capacity) for _ in range(POP_SIZE // 2)]
        offsprings = nextGA(parents, POP_SIZE - len(parents), jobs, MUTATION_RATE)
        population = parents + offsprings

    return 1 / best_score, best_solution, total_history, total_history.index(min(total_history)), best_step_sequence

def generate_gantt_chart(job_steps, machine_capacity):
    """繪製甘特圖，考慮機器容量，避免同時安排超過容量的工作"""
    base_time = datetime(2025, 5, 1)
    machine_color_map = {}
    colors = plt.cm.get_cmap('tab20', 20)
    color_index = 0

    gantt_data = []
    machine_timeline = defaultdict(list)  # 每台機器各個 slot 的占用時間列表
    slot_assignment = {}  # (job_id, step_index) -> slot number
    end_time=0
    job_time_range = {}

    for job_id, job_info in job_steps.items():
        start_time = base_time
        for step_index, step in enumerate(job_info["step_sequence"]):
            machine = step["machine"]
            duration = timedelta(hours=step["processing_time"])
            end_time = start_time + duration

            # 初始化機器顏色
            if machine not in machine_color_map:
                machine_color_map[machine] = colors(color_index)
                color_index += 1

            # 找到一個有空閒時間的 slot
            capacity = machine_capacity.get(machine, 1)
            assigned_slot = None
            for slot in range(capacity):
                timeline = machine_timeline[(machine, slot)]
                if not timeline or timeline[-1][1] <= start_time:
                    assigned_slot = slot
                    timeline.append((start_time, end_time))
                    break

            # 如果沒有可用 slot，延後這個 step 的執行時間直到有空（簡單模擬 FIFO 等待）
            if assigned_slot is None:
                soonest_slot, soonest_time = None, datetime.max
                for slot in range(capacity):
                    last_end = machine_timeline[(machine, slot)][-1][1]
                    if last_end < soonest_time:
                        soonest_time = last_end
                        soonest_slot = slot
                start_time = soonest_time
                end_time = start_time + duration
                assigned_slot = soonest_slot
                machine_timeline[(machine, assigned_slot)].append((start_time, end_time))

            slot_assignment[(job_id, step_index)] = assigned_slot

            #更新時間:
            if job_id not in job_time_range:
                job_time_range[job_id] = [start_time, end_time]
            else:
                job_time_range[job_id][0] = min(job_time_range[job_id][0], start_time)
                job_time_range[job_id][1] = max(job_time_range[job_id][1], end_time)

            gantt_data.append({
                "label": f"{machine}-{assigned_slot+1}",
                "job": f"Job {job_id}",
                "start": start_time,
                "duration": duration,
                "machine": machine,
                "color": machine_color_map[machine]
            })

            start_time = end_time  # 下一步從這裡繼續
    job_durations = {
        job_id: round((end - start).total_seconds() / 3600, 2)
        for job_id, (start, end) in job_time_range.items()
    }

    # 繪圖
    fig, ax = plt.subplots(figsize=(14, 6))
    y_labels = []
    y_ticks = []
    label_to_y = {}
    current_y = 10
    height = 8

    # 先排序 label，讓相同機器類型靠近
    sorted_labels = sorted(set(entry["label"] for entry in gantt_data), key=lambda x: (x.split('-')[0], int(x.split('-')[1])))

    for label in sorted_labels:
        label_to_y[label] = current_y
        y_ticks.append(current_y + height / 2)
        y_labels.append(label)
        current_y += 15

    # 畫上 bar
    for entry in gantt_data:
        label = entry["label"]
        y_pos = label_to_y[label]
        
        start_hour = (entry["start"] - base_time).total_seconds() / 3600
        duration_hr = entry["duration"].total_seconds() / 3600
        ax.broken_barh([(start_hour, duration_hr)], (y_pos, height), facecolors=entry["color"])
        ax.text(start_hour + duration_hr / 2, y_pos + height / 2, entry["job"], 
                ha='center', va='center', color='white', fontsize=8)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Hours since start")
    ax.set_title("Gantt Chart - Machine Scheduling (With Capacity)")
    # 加入淺色橫向網格線
    ax.grid(True, which='major', axis='both', linestyle='--', color='lightgray', linewidth=0.5)
    ax.set_axisbelow(True)  # 確保網格在線條底下

    # 圖例
    legend_handles = [mpatches.Patch(color=color, label=machine) for machine, color in machine_color_map.items()]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    matplotlib.use('Agg')

    # 轉 base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    gantt_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return gantt_base64,job_durations,end_time




def run_ga(jobs, job_deadlines, machine_times, machine_capacity):
    """ 執行 GA 並生成甘特圖 """

    start_time = time.time()
    POP_SIZE, GENS, MUTATION_RATE = 80, 20, round(random.random(), 3)
    machine_list = list(set(machine for job in jobs.values() for machine in job))

    best_steps, best_solution, fit_history, min_gens, best_step_sequence = GA(
        jobs, job_deadlines, POP_SIZE, GENS, MUTATION_RATE, machine_list, machine_times, machine_capacity
    )

    runtime = round(time.time() - start_time, 2)

    job_steps = {
        job_id: {
            "steps": len(best_step_sequence.get(job_id, [])),
            "completion_time": sum(step["processing_time"] for step in best_step_sequence.get(job_id, [])),
            "step_sequence": best_step_sequence.get(job_id, [])
        }
        for job_id in jobs.keys()
    }

    gantt_chart,job_durations,end_time = generate_gantt_chart(job_steps, machine_capacity)
    print(job_durations)

    matplotlib.use('Agg') 
    # 繪製收斂圖
    buf = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(fit_history, color='red')
    plt.title(f'Convergence Graph - mutate: {MUTATION_RATE}, popsize: {POP_SIZE}, time={runtime}s')
    plt.xlabel('Generation')
    plt.ylabel('Step')
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close("all")

    return {
        "best_steps": best_steps,
        "runtime": runtime,
        "min_gens": min_gens,
        "plot": img_base64,
        "gantt_chart": gantt_chart,
        "job_steps": job_steps,
        "job_durations":job_durations,
        "end_time":end_time
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)