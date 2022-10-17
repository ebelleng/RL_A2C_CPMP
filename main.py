from actor_critic import *
from greedy import *
from data import generate_data_a2c
from enviroment import Enviroment
import time
import matplotlib.pyplot as plt

def show_time(time):
    if time>60 and time<3600: 
        return f'{(time/60):.0f} minutos'
    elif time>3599:          
        return f'{time/3600:0f} horas'
    return f'{time:.0f} segundos'    

def print_report(a2c_steps,greedy_steps):
    print('---------------------------------')
    print("Greedy: ",greedy_steps)
    print("Pasos: ",len(greedy_steps))
    print('---------------------------------')
    print("A2C   : ",a2c_steps)
    print("Pasos: ",len(a2c_steps))
    print()

def save_report(state_initial,a2c_steps,greedy_steps,H=7,S=7):
    with open(f'steps_report_{S}x{H}.csv','a') as f:
        f.write(f'state_initial  : {state_initial}\n')
        lay = [ fila + [0]*(H-len(fila)) for fila in state_initial ]

        for i in range(H-1, -1, -1):
            for j in range(len(lay)):
                f.write(f'{lay[j][i]:2}  ')
            f.write('\n')
        f.write('\n')
        f.write(f'actions_a2c    : {a2c_steps}\n')
        f.write(f'actions_greedy : {greedy_steps}\n')
        f.write(f'steps_a2c      : {len(a2c_steps)}\n')
        f.write(f'steps_greedy   : {len(greedy_steps)}\n')
        f.write('\n')
    return

def guardar_graficas(pasos,pasos_g,n_iter,N,rows,columns):
    plt.figure(0)
    plt.title(f"Entrenamiento con N=[1,{N}] para {rows}x{columns}", size=16)
    plt.rcParams["figure.figsize"] = (20,15)


    plt.xlabel("Cantidad problemas", size = 12,)
    plt.ylabel("Cantidad de pasos", size = 12)

    X_plot = range(len(pasos))
    y_plot = pasos
    plt.plot(X_plot, y_plot, label='A2C')

    plt.vlines(x = [n for n in range(n_iter,N*n_iter,n_iter)],ymin = 0, ymax = max(y_plot), 
           colors = 'grey', 
           linestyles='--')

    X_plot = range(len(pasos_g))
    y_plot = pasos_g
    plt.plot(X_plot, y_plot, label='Greedy')
    plt.legend(loc='lower right')

    plt.savefig(f'img\\training_{rows}x{columns}.png')
    plt.close()

    plt.figure(1)
    plt.title(f"Entrenamiento con N=[1,10]", size=16)
    plt.rcParams["figure.figsize"] = (20,15)

    plt.boxplot([pasos,pasos_g])

    plt.savefig(f'img\\train_boxplot_{rows}x{columns}.png')
    plt.close()
    
def main():
    rows,columns = 7,7
    env          = Enviroment(rows,columns)
    actor_critic = ActorCritic(env)
    X_train,X_test,y_train,y_test  = generate_data_a2c(10,actor_critic,rows,columns)
    
    # Entrenamiento
    actor_critic.fit(X_train,y_train,verbose=True,save_files=True)
    y_pred       = actor_critic.predict(X_test) ; y_pred.to_csv('y_pred.csv',index=None)

    # Metricas
    #actor_critic.actor_metrics(y_test['actions'], y_pred['actions'])
    #actor_critic.critic_metrics(y_test['rewards'], y_pred['rewards'])
    
def _main(MaxIter=1000,n_pasos=8):
    for n in range(3,8):
        rows,columns = n,n
        env = Enviroment(rows,columns)
        actor_critic = ActorCritic(env)
        
        y_data_a2c    = []
        y_data_greedy = []

        for N in range(1,n_pasos+1):
            greedy_pasos = []
            a2c_pasos    = []

            print(f'#    Problemas para N={N}    #')
            for i in range(MaxIter):
                layout = env.create_env(n**2 - (n**2)//2)
                print(f'#            {i}             #')
                #print(layout.stacks)

                greedy_actions = greedy_solve(copy.deepcopy(layout))
                a2c_actions = actor_critic.solve(copy.deepcopy(layout), greedy_solve,train=True,n_pasos=N)
                
                #print_report(a2c_actions,greedy_actions)
                save_report(layout.stacks,a2c_actions,greedy_actions,rows,columns)

                greedy_pasos.append(len(greedy_actions))
                a2c_pasos.append(len(a2c_actions))

            y_data_a2c += a2c_pasos
            y_data_greedy += greedy_pasos

            print(f'# ------------------------ #')
            print(f'# Cantidad de problemas: {MaxIter} ')
            print(f"# Greedy prom: {sum(greedy_pasos)/MaxIter:.1f} ")
            print(f"# A2c prom: {sum(a2c_pasos)/MaxIter:.1f} ")
            print('# ------------------------ #')

        guardar_graficas(y_data_a2c,y_data_greedy,MaxIter,N,rows,columns)
        #actor_critic.graph()
        return

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f'Tiempo de ejecución: {show_time(end-start)}')
    
