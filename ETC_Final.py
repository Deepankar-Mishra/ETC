import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math
def ETC(K,n,m,delta,u_star_list):
    m_min=1
    m_max=int(n/K)
    empirical_means=[0 for i in range(K)]
    num_of_pulls=[0 for i in range(K)]
    optimal_means=u_star_list[:]
    u_opt=max(optimal_means)
    act_regret=0
    emp_regret=0
    #Explore
    for i in range(m):
        for j in range(K):
            reward=np.random.normal(optimal_means[j], 1)
            emp_regret=u_opt-empirical_means[j]
            empirical_means[j]=(reward+num_of_pulls[j]*empirical_means[j])/(1+num_of_pulls[j])
            num_of_pulls[j]+=1
            act_regret+=u_opt-optimal_means[j]
    #Exploit
    exploit_arm_number=empirical_means.index(max(empirical_means))
    for i in range(m*K,n+1):
        reward=np.random.normal(empirical_means[exploit_arm_number], 1)
        emp_regret=u_opt-empirical_means[exploit_arm_number]
        empirical_means[j]=(reward+num_of_pulls[exploit_arm_number]*empirical_means[exploit_arm_number])/(1+num_of_pulls[exploit_arm_number])
        num_of_pulls[exploit_arm_number]+=1
        act_regret+=u_opt-optimal_means[exploit_arm_number]
    return(act_regret,emp_regret)

K=2
n=1000
m_min=1
m_max=int(n/K)
delta_values = np.arange(0.01, 1.01, 0.01)
actual_regret=np.zeros_like(np.arange(0.01, 1.01, 0.01))
emp_regret=np.zeros_like(np.arange(0.01, 1.01, 0.01))
theoretical_bounds = []
for delta in delta_values:
    term1=n*delta
    term2=math.log((n*delta*delta)/4)
    term2=max(0,term2)
    term2+=1
    term2=(4*term2)/delta
    term2+=delta
    theoretical_bounds.append(min(term1,term2))
for ep in tqdm(range(100000)):
    ep_actual_regret=[]
    ep_emp_regret=[]
    delta_list=[]
    for delta in delta_values:
        m_opt=max(1, int(4 / delta**2 * np.log(n * delta**2 / 4)))
        u_star_list=[0,-1*delta]
        cummulative_act_regret,cummulative_emp_regret=ETC(K,n,m_opt,delta,u_star_list)
        ep_actual_regret.append(cummulative_act_regret)
        ep_emp_regret.append(cummulative_emp_regret)
        delta_list.append(delta)
    actual_regret=(ep*actual_regret+ep_actual_regret)/(ep+1)
    emp_regret=(ep*emp_regret+ep_emp_regret)/(ep+1)                
plt.figure(figsize=(10, 6))
plt.plot(delta_values, actual_regret, label="Expected Regret", color="blue")
plt.plot(delta_values, theoretical_bounds, label="Theoretical Upper Bound", color="red")
#plt.plot(delta_list, theoretical_bounds, label="Theoretical Upper Bound")
plt.xlabel("Delta")
plt.ylabel("Expected Regret")
plt.legend()
plt.title("Expected Regret vs. Delta of ETC Algorithm")
#plt.grid(True)
plt.show()

    