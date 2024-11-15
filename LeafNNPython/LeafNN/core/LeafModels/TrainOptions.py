class TrainOption:
    def __init__(self):
        self.MaxIteration = 100
        #lineSearch Option
        self.MaxLineSearch = 20
        self.C1 = 0.01 # line Search: wolfe conditon 1
        self.C2 = 0.5  # line search: wolfe condtion 2
        self.INT = 0.1 
        self.EXT = 3.0
        self.RATIO = 100
        self.learnRate = 0.001
        # regularization part 
        self.regularEnable = True
        self.regularLamada = 0.0

        #early stop
        self.enableEarlyStop = False
        self.ESTFrequency = 100
        self.ESTminiDiffCost = 0.01
        #train option: gradient check option
        self.enableGradientCheck = False 
        self.gradientCheckFrequency = 100
        # train/validation/test data ratio
        self.trainRatio = 0.7
        self.validationRatio = 0.15
        self.testRatio = 0.15
