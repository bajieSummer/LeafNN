import matplotlib.pyplot as plt
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.Log import Log
tag_msg = "PlotNewtonHelper"
class PlotNewtonHelper:
    def getContourMesh(x0_min,x0_max,x1_min,x1_max,func,argsList,steps=400):
        if x0_max == x0_min: x0_max +=1
        if x1_max == x1_min: x1_max +=1
        # make the shape not scale
        x0_min =min(x0_min,x1_min)
        x1_min =x0_min
        x0_max = max(x0_max,x1_max)
        x1_max = x0_max
        fmin = None 
        fmax = None

        X0_l = MM.linspace(x0_min, x0_max, steps)
        steps1 =int(steps*(x1_max-x1_min)/(x0_max-x0_min))
        X1_l = MM.linspace(x1_min, x1_max, steps1)
        X0, X1 = MM.meshgrid(X0_l, X1_l)
        Z =  MM.zeros([steps1,steps])
        for i in range(steps1):
            for j in range(steps):
                xi = MM.array([[X0[0,j]],[X1[i,0]]])
                Z[i,j] = func(xi,argsList)
                if fmin == None: fmin ,fmax=Z[i,j],Z[i,j]
                fmin=min(fmin,Z[i,j])
                fmax=max(fmax,Z[i,j])
        return (X0,X1,Z,fmin,fmax)


    def drawHisData(hisData,extend=2,calPolyF=None,argsList=None,f_str=""):
        X0data =[]
        X1data = []
        X0_min = None
        X0_max = None 
        X1_min = None 
        X1_max = None
        dimension = len(hisData[0][0])
        for d in hisData:
            X = d[0]
            f = d[1]
            grad =d[2]
            x0 = X[0,0]
            x1 = f
            if dimension>1:x1 = X[1,0]
            X0data.append(x0)
            X1data.append(x1)
            if X0_min is None:X0_min,X0_max =x0,x0
            X0_min = min(x0,X0_min)
            X0_max = max(x0,X0_max)
            if X1_min is None:X1_min,X1_max =x1,x1
            X1_min = min(x1,X1_min)
            X1_max = max(x1,X1_max)
        label ='x0_x1'

        if dimension ==1:
            label='XY'
        #draw x0x1 points and draw x0x1 line

    
        iterNum=len(X0data)
        # draw contour lines
        if dimension >1 and calPolyF is not None:
            (Xt,Yt,Zt,fmin,fmax) = PlotNewtonHelper.getContourMesh(X0_min-extend,X0_max+extend,X1_min-extend,X1_max+extend,calPolyF,argsList)
            #Log.Debug(tag_msg,f"Xt=\n{Xt}\n,Yt=\n{Yt}\n,Zt=\n{Zt}\n")
            Log.Debug(tag_msg,f"getMinMax_from CoutourMap fmin={fmin},fmax={fmax}\n")
            contour = plt.contourf(Xt, Yt, Zt, levels=MM.linspace(fmin, fmax, 30), cmap='viridis')
            ###plt.contour(Xt, Yt, Zt, levels=10, linestyles='dashed', colors='black')
            cbar = plt.colorbar(contour)
            cbar.set_label('Z Value')
        else: # draw function line
            xt = MM.linspace(X0_min-extend,X0_max+extend,num=200)
            yt = []
            for xti in xt: yt.append(calPolyF(MM.array([[xti]]),argsList))
            plt.plot(xt,yt,color='blue',linestyle="--", label=label)

        plt.scatter(X0data, X1data, color='red', s=5, marker='D', label=label)
        plt.plot(X0data,X1data,color='black',label=label)
        plt.title(f"Draw Iters:f={f_str},convergeNum={iterNum}")
        plt.show()
    
