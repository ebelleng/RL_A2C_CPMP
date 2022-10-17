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
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f'Tiempo de ejecución: {show_time(end-start)}')
    
