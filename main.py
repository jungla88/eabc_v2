from Extractor import Extractor
from Granulator import Granulator
from Agent import Agent
from Metric import Metric
from Representative import Representative
from Clustering_MBSAS import Clustering_MBSAS
from Clustering_K_Means import Clustering_K_Means

extractor1 = Extractor()

obj_clustering_MBSAS = Clustering_MBSAS(3, 0.2, 0.1, 1.1) # Lambda, theta_start ,theta_step, theta_stop
agent1 = Agent(Granulator, Metric, extractor1, Representative, obj_clustering_MBSAS)
agent1.execute(3.1,0.5) # S_T, eta

obj_clustering_K_Means = Clustering_K_Means(1,3) #k, k_max
agent2 = Agent(Granulator, Metric, extractor1, Representative, obj_clustering_K_Means)
agent2.execute(3.1,0.5) # S_T,  eta
