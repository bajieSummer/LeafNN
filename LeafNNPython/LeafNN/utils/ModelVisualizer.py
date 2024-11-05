import numpy as np
import matplotlib.pyplot as plt
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf

class ModelVisualizer:
    """
    only for two layers network
    """
    def plot2DDecisionBoundary(wb:NeuralLeaf,X,Y):
        # boundary h(z(x,cita)) > 0.5 ->1.0 h(z(x,cita))<0.5 -->0.0
        # z(x,cita) >0 -->1.0 z(x,cita)<0 -->0.0 z(x,cita) = 0 -->boundary
        # first get dataX(x0->max,->min) z(x1,x2) = 0 ->x2min,x2max
        # Separate the data based on the values of y
        # first dimension
        inds0 = 0
        # second dimension
        inds1 = 1
            
        x_y0 = X[Y.flatten() == 0]
        x_y1 = X[Y.flatten() == 1]

        # Plot the points with different shapes or colors based on the value of y
        plt.scatter(x_y0[:, inds0], x_y0[:, inds1], color='blue', marker='s', label='y=0')  # Squares for y=0
        plt.scatter(x_y1[:, inds0], x_y1[:, inds1], color='blue', marker='^', label='y=1')   # Triangles for y=1
        x0_max = X[0][inds0]
        x0_min = X[0][inds0]
        for xi in X:
            if(xi[inds0]>x0_max):
                x0_max = xi[inds0]
            if(xi[inds0]<x0_min):
                x0_min = xi[inds0]
        # wx+b = 0 b*1.0 + w1*x1+w2*x2 =0 
        # todo not consider w1*x1 + w2*x2 + w3*x3 ...=0 such situation
        x1_max = (x0_max*wb[0][1][0] + +wb[0][0][0])/(-1.0*(wb[0][2][0]+np.finfo(float).eps))
        x1_min = (x0_min*wb[0][1][0] + +wb[0][0][0])/(-1.0*(wb[0][2][0]+np.finfo(float).eps))
        x0min_max =[x0_min,x0_max]
        x1min_max =[x1_min,x1_max] 
        plt.plot(x0min_max,x1min_max,linestyle="-")
        
        # Add labels and title
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('x has 2 features. classification plotting')

        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()

    

    def plotData(X,Y):
        # Separate the data based on the values of y
        x_y0 = X[Y.flatten() == 0]
        x_y1 = X[Y.flatten() == 1]

        # Plot the points with different shapes or colors based on the value of y
        plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
        plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1

        # Add labels and title
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('x has 2 features. classification plotting')

        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()


    def plotDataWithTestCase(X,Y,xt,yt_p,yt):
        # Separate the data based on the values of y
        x_y0 = X[Y.flatten() == 0]
        x_y1 = X[Y.flatten() == 1]

        # Plot the points with different shapes or colors based on the value of y
        plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
        plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1

        markert = 's'
        if yt_p > 0.5 : markert ='^'
        colort = 'green'
        print(f"yt_p={yt_p},yt={yt}")
        if(np.abs(yt_p - yt)>0.5): colort='red'
        plt.scatter(xt[:, 0], xt[:, 1], color=colort, marker=markert)  
        
        
        # Add labels and title
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('x has 2 features. classification plotting')

        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()

    def plotCostWithWB(inds,costs,wb=None):
        length = len(costs)
        if length <= 0:
            print("error:costs is empty")
            return

        lastCost = costs[length-1]
        title = f"cost/iterations lastCost={lastCost}"
        if(wb is not None):
            title=f"{title},initW={wb[0][0]},initB={wb[1][0]}"
        plt.title(title)
        plt.scatter(inds,costs,color='orange',marker='s',label ='cost')
        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()

    def plotCost(inds,costs):
        ModelVisualizer.plotCostWithWB(inds,costs)

   