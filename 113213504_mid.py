import random
import matplotlib.pyplot as plt
import time
#作者: 11321350 林千榆
#高等生產管理 mid
# 流程：每個工件（job）需要經過的站點（machine）

# 機台編號對應的實際名稱
real_seq = {1: 'R', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E', 7: 'F', 8: 'S'}


#"""
num_jobs=5
jobs = {
    1: [1, 2, 3, 5, 4, 7, 8],
    2: [1, 3, 5, 4, 2, 8],
    3: [1, 6, 7, 3, 2, 4, 5, 8],
    4: [1, 7, 2, 4, 5, 8],
    5: [1, 4, 2, 5, 8]
}
less_step=35
"""
jobs = {}

num_jobs = int(input("請輸入物品總數（job 數量）："))
print("請以數字代替加工步驟 1: 'R', 2: 'A', 3: 'B', 4: 'C', 5: 'D', 6: 'E', 7: 'F', 8: 'S'}")
print("**此程式沒有防呆機制，請不要輸入空白，且每個工作請至少輸入一個工作站**")
less_step=0
for i in range(1, num_jobs + 1):
    route_input = input(f"請輸入第 {i} 個物品所需經過的工作站（以逗號分隔）：")

    # 將輸入的字串轉成整數 list，例如 "1, 3,5" → [1, 3, 5]
    stations = list(map(int, route_input.strip().split(',')))
    #計算中間間隔 3->1=2 5->3=2 最少步數為4
    less_step=num_jobs*7
    jobs[i] = stations

#"""
# 初始化種群：隨機排列，但將 R 永遠放在第一
def initialPop(POP_SIZE):
    population = []
    for _ in range(POP_SIZE):
        order = list(range(1, 9))  # 站點
        random.shuffle(order)  # 隨機排列
        population.append(order)
    return population

# 計算加工步驟的總移動成本
def stepCA(schedule, jobs):
    total_steps = 0
    job_steps = {}

    for job_id, seq in jobs.items():
        steps = 0
        step_order = []
        for i in range(1, len(seq)):
            if seq[i] in schedule and seq[i - 1] in schedule:
                step_count = abs(schedule.index(seq[i]) - schedule.index(seq[i - 1]))
                steps += step_count
                step_order.append((real_seq[seq[i - 1]], real_seq[seq[i]], step_count))      
        job_steps[job_id] = (steps, step_order)
        total_steps += steps
    return total_steps, job_steps

# 適應度函數：步驟越少越好
def efficiency(schedule, jobs):
    total_steps, _ = stepCA(schedule, jobs)
    return less_step/total_steps   # 步驟少的適應度高

# 選擇父代（竟賽選擇）
def selectGA(population, jobs):
    #在樣本中隨機選取五個，並挑選最佳
    return max(random.sample(population, 5), key=lambda ind: efficiency(ind, jobs))
"""
def selectGA(population, jobs):
    # 計算每個個體的適應度
    fitnesses = [efficiency(ind, jobs) for ind in population]

    # 將負值轉換為正值（如果 efficiency 有可能小於 0）
    min_fit = min(fitnesses)
    if min_fit < 0:
        fitnesses = [f - min_fit + 1e-6 for f in fitnesses]  # 加個小常數避免除以 0

    # 計算總適應度與機率分佈
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]

    # 使用權重隨機抽選一個個體（類似轉輪盤）
    selected = random.choices(population, weights=probs, k=1)[0]
    return selected
"""
# 交叉
def crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(0, size), 2)) 
    child = [-1] * size
    child[p1:p2] = parent1[p1:p2]

    remaining = [gene for gene in parent2 if gene not in child]
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining.pop(0)
    return child

# 突變：隨機交換兩個機台
def mutate(offspring, MUTATION_RATE):
    if random.random() < MUTATION_RATE:
        #i, j = random.sample(range(0, len(offspring)), 2) #swapping 
        i, j = sorted(random.sample(range(0, len(offspring)), 2))  # inversion 
        #offspring[i], offspring[j] = offspring[j], offspring[i]
        """"""
        #scramble 
        sub = offspring[i:j]
        random.shuffle(sub)
        offspring[i:j] = sub
        
    return offspring


# 產生子代
def nextGA(parents, pop_size, jobs, MUTATION_RATE):
    offsprings = []
    while len(offsprings) < pop_size:
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child, MUTATION_RATE)
        offsprings.append(child)
    return offsprings

# 基因演算法（GA）
def GA(jobs, POP_SIZE, GENS, MUTATION_RATE):
    #初始化族群
    population = initialPop(POP_SIZE)
    min_time=0
    best_steps = float('inf')
    total_history=[]
    #tmp=0

    for i in range(GENS):
        #計算效率
        efficiency_value = [efficiency(index, jobs) for index in population]

        #挑選最佳適應值
        best_index = efficiency_value.index(max(efficiency_value))  
        best_candidate = population[best_index]
        total_steps, job_steps = stepCA(best_candidate, jobs)
        total_history.append(total_steps)

        #取最佳解
        if total_steps < best_steps:
            best_steps = total_steps
            min_time=i
        #基因更迭
        parents = [selectGA(population, jobs) for j in range(POP_SIZE // 2)]
        offsprings = nextGA(parents, POP_SIZE - len(parents), jobs, MUTATION_RATE)
        population = parents + offsprings
    return  best_steps, job_steps ,total_history,min_time


def main():
    start = time.time()
    POP_SIZE = 80   # 種群大小
    GENS = 20      # 迭代次數
    MUTATION_RATE = float(round(random.random(), 3)) # 突變率



    best_steps, job_steps ,fit,min_gens= GA(jobs, POP_SIZE, GENS, MUTATION_RATE)
    end = time.time()
    runtime=round(end-start,2)
    # 格式化最佳加工順序
    print("\n各個站的加工步數及步驟:")
    for job_id, (steps, step_order) in job_steps.items():
        print(f"工作 {job_id}:")
        step_tmp=[]
        for step in step_order:
            step_tmp.append(step[0])
        step_tmp.append(step_order[-1][1])
        print(f"步驟:{step_tmp}")
        print(f"總步數: {steps}")
    print(" 最佳步數為:", best_steps)
    print(f"Efficiency = {less_step}/{best_steps} : {less_step / best_steps:.2f}")
    print("最佳次數:",min_gens)
    print(f"耗{runtime} 秒")
    plt.figure(figsize=(10, 6))
    plt.plot(fit, color='red')
    plt.title(f'Convergence Graph - mutate: {MUTATION_RATE}, popsize:{POP_SIZE}, time={runtime}')
    plt.xlabel('Generation')
    plt.ylabel('step')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
