from Granule import Granule
import matplotlib.pyplot as plt
from statistics import mean

class Granulator:
    lista_di_granuli = [];
    
    def __init__(self, obj_metric, obj_representative, obj_clustering, S_T,eta):
        self.obj_metric = obj_metric
        self.obj_representative = obj_representative 
        self.obj_clustering = obj_clustering
        self.S_T = S_T
        self.eta = eta

    def Process(self, dataset):
        
        representatives, clusters_v = self.obj_clustering.evaluate(dataset, self.obj_metric, self.obj_representative)
        self.Add(representatives, clusters_v)
         
        return representatives, clusters_v, self.lista_di_granuli
    
    def Add(self,representatives,clusters_v):
        
        # Calcolo cardinalità
        cardinalita = []
        for i in range(0,len(clusters_v)):
            cardinalita.append(len(clusters_v[i]))
        
        # Calcolo distanze dei punti dei clusters dai loro rappresentanti
        distanze = []
        for i in range(0,len(representatives)):
            distanze.append([])
            for j in range(1,len(clusters_v[i])):
                distanza = self.obj_metric.Diss(clusters_v[i][j],representatives[i])
                distanze[i].append(distanza)
                
        # Calcolo compattezza
        compattezza = []
        for i in range(0,len(representatives)):
            somma = 0
            end = len(distanze[i])-1
            for j in range(0,len(distanze[i])):
                somma = somma + distanze[i][j] # Somma di tutte le distanze di un cluster
                if j == end:
                    compattezza.append(somma)
        
        # Calcolo Effective Radius
        effective_Radius = []
        for i in range(0, len(cardinalita)):
            avg = mean(distanze[i])
            effective_Radius.append(avg)    
        
        # Calcolo Quality (valor medio di compattezza e cardinalità)
        quality = []
        for i in range(0, len(cardinalita)):
            avg2 = (compattezza[i]+cardinalita[i])*self.eta
            quality.append(avg2)
        
        
        print("Quality")
        print(quality)
        
        for quality2 in quality:
            if quality2 > self.S_T:
                # Creazione oggetto granulo
                granulo = Granule()
                # Set Params oggetto granulo
                granulo.set_Representative(representatives)
                # Assegno cardinalità
                granulo.set_Cardinality(cardinalita)
                # Set di compattezza
                granulo.set_Compactness(compattezza)
                # Set Effective Radius
                granulo.set_Compactness(effective_Radius)
                # Set Quality
                granulo.set_Quality(quality2)
                # Inserimento in lista oggetto granulo
                self.lista_di_granuli.append(granulo)
        

    
       
    
        

