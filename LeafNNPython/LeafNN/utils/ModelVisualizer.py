from LeafNN.Bases.MathMatrix import MathMatrix as MM
import matplotlib.pyplot as plt
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf
from LeafNN.utils.Log import Log
#np.array = MM.original_np_array
class ModelVisualizer:
    """
    only for two layers network
    """
    def plot2DDecisionBoundary(wb:NeuralLeaf,X,Y,meshX=None,meshY=None, boundaryValues=None,Single=False):
        # boundary h(z(x,cita)) > 0.5 ->1.0 h(z(x,cita))<0.5 -->0.0
        # z(x,cita) >0 -->1.0 z(x,cita)<0 -->0.0 z(x,cita) = 0 -->boundary
        # first get dataX(x0->max,->min) z(x1,x2) = 0 ->x2min,x2max
        # Separate the data based on the values of y
        # first dimension
        inds0 = 0
        # second dimension
        inds1 = 1
        [m,n]=X.shape    
        x_y0 = X[Y.flatten() == 0]
        x_y1 = X[Y.flatten() == 1]

        if Single:
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
        if(n<=2):
            if(n<2):
                Log.Error("ModelVisualize",f"X only have{n} features, not enough to support decision boundary yet")
                return
            x1_max = (x0_max*wb[0][1][0] + +wb[0][0][0])/(-1.0*(wb[0][2][0]+MM.finfo().eps))
            x1_min = (x0_min*wb[0][1][0] + +wb[0][0][0])/(-1.0*(wb[0][2][0]+MM.finfo().eps))
            x0min_max =[x0_min,x0_max]
            x1min_max =[x1_min,x1_max] 
            plt.plot(x0min_max,x1min_max,linestyle="-")
        else:
            plt.contour(meshX[:,inds0], meshX[:,inds1], meshY, levels=boundaryValues, colors='blue', linewidths=2)
        if Single:
            # Add labels and title
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('x has 2 features. classification plotting')

            # Add a legend
            plt.legend()
            # Display the plot
            plt.show()

    def plotData(X,Y,titleAdd=None):
        # Separate the data based on the values of y
        x_y0 = X[Y.flatten() == 0]
        x_y1 = X[Y.flatten() == 1]

        # Plot the points with different shapes or colors based on the value of y
        
        plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
        plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1

        # Add labels and title
        plt.xlabel('X1')
        plt.ylabel('X2')
        if titleAdd is None:
            titleAdd = ""
        title = f"X shape={X.shape},we plot 2 features,{titleAdd}"
        plt.title(title)

        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()


    def plotDataWithTestCase(X,Y,xt,yt,yt_p):
        # Separate the data based on the values of y
        x_y0 = X[Y.flatten() == 0]
        x_y1 = X[Y.flatten() == 1]

        # Plot the points with different shapes or colors based on the value of y
        plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
        plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1

        i = 0
        noPass= 0
        for xi in xt:
            markert = 's'
            if yt[i] > 0.5 : markert ='^'
            colort = 'green'
            if(MM.abs(yt_p[i] - yt[i])>=0.5):
                colort='red'
                noPass +=1
            plt.scatter(xt[i,0], xt[i, 1], color=colort, marker=markert) 
            i+=1 
        # Add labels and title
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(f'TestCase TrainNums={len(X)}\ntrainTest:nums={i},passeRate={(i-noPass)/i}')

        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()

    def plotCostWithWB(inds,costs,initWB=None,newWB=None):
        length = len(costs)
        if length <= 0:
            print("error:costs is empty")
            return

        lastCost = costs[length-1]
        title = f"lastC={lastCost:.4f}"
        if(initWB is not None):
            title=f"{title},initbw0={initWB[0][0][0]:.2f},{initWB[0][1][0]:.2f}"
        if(newWB is not None):
            title = f"{title},newbw0={newWB[0][0][0]:.2f},{newWB[0][1][0]:.2f}"
        plt.title(title)
        plt.scatter(inds,costs,color='orange',marker='s',label ='cost')
        plt.xlabel('iteration')
        plt.ylabel('cost')
        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()

    def plotCost(inds,costs):
        ModelVisualizer.plotCostWithWB(inds,costs)

    def plotYYpre(inds,Y,Y_pre,title):
        Y_inds = Y[inds,:]
        Y_inds_pre = Y_pre[inds,:]

        # Plot the points with different shapes or colors based on the value of y
        plt.scatter(inds, Y_inds, color='green', marker='^', label='from data')  # Squares for y=0
        plt.scatter(inds, Y_inds_pre, color='orange', marker='v', label='from predict')   # Tria
        plt.title(title)
        plt.xlabel('indices')
        plt.ylabel('Y')
        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()
   
    def plotGradientsSquare(inds,gradients,title):
        plt.title(title)
        gradSqaure = []
        for gd in gradients:
            gradSqaure.append(gd*gd)
        plt.scatter(inds,gradSqaure,color='orange',marker='s',label ='GradientsStrength')
        Log.Debug("gradientsStrength>>>",gradSqaure)
        plt.xlabel('iteration')
        plt.ylabel('gradientsSquare')
        # Add a legend
        plt.legend()
        # Display the plot
        plt.show()