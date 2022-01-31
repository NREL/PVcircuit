
    ####################
    # following only for single junction as a cell
    # better to using multi2T with one junction
    ####################

    def Jdiode(self,Vtot):
        #return J for Vtot
        #only used for single junction cell
        return  self.Jparallel(self.Vmid(Vtot),self.Jphoto) * self.totalarea / self.lightarea    


    def Jcell(self,Vcell):
        #Vcell, Jcell are iterable

        Jcell=[]    #empty list
        
        for Vtot in Vcell:
             Jcell.append(-self.Jdiode(Vtot))
          
        return Jcell

    def Vcell(self,Jcell):
        #Vcell, Jcell are iterable

        Vcell=[]    #empty list
        
        for Jdiode in Jcell:
            try:
                Vtot=self.Vdiode(Jdiode) + Jdiode * self.Rser
            except:
                Vtot=None
                
            Vcell.append(Vtot) 
          
        return Vcell

    @property
    def Voc(self):
        #Jdiode=0
        return self.Vdiode(0.) 
        
    @property
    def Jsc(self):
        #Vtot=0
        return self.Jdiode(0.)
    
    @property
    def MPP(self):
        # calculate maximum power point and associated IV, Vmp, Jmp, FF
        
        ts = time()
        res=0.001   #voltage resolution
        self.JLC = 0.
        if self.Jphoto > 0.: #light JV
            self.Vsingle = list(np.arange(-0.2, (self.Voc*1.02), res))
            self.Jsingle = self.Jcell(self.Vsingle)
            self.Psingle = [(-v*j) for v, j in zip(self.Vsingle,self.Jsingle)]
            nmax = np.argmax(self.Psingle)
            self.Vmp = self.Vsingle[nmax]
            self.Jmp = self.Jsingle[nmax]
            self.Pmp = self.Psingle[nmax]
            self.FF = abs((self.Vmp * self.Jmp) / (self.Voc * self.Jsc))
            
            self.Vpoints = [0., self.Vmp, self.Voc]
            self.Jpoints = [-self.Jsc, self.Jmp, 0.]
            
        else:   #dark JV
            self.Jsingle = list(np.logspace(-13., 7., num=((13+7)*3+1)))
            self.Vsingle = self.Vcell(self.Jsingle)
            self.Vmp = None
            self.Jmp = None
            self.Pmp = None
            self.FF = None
            self.Vpoints = None
            self.Jpoints = None
 
        mpp_dict = {"Voc":self.Voc, "Jsc":self.Jsc, \
                    "Vmp":self.Vmp,"Jmp":self.Jmp, \
                    "Pmp":self.Pmp, "FF":self.FF}
                        
        te = time()
        ms=(te-ts)*1000.
        #print(f'MPP: {res:g}V , {ms:2.4f} ms')
        
        return mpp_dict
                                            
    def plot(self,title=None):
        #plot a single junction
        
        if self.name:
            title = self.name + title
                       
        self.MPP   #generate IV curve and analysis
        
        fig, ax = plt.subplots()
        ax.plot(self.Vsingle,self.Jsingle)  #JV curve
        ax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
        ax.set_ylabel('Current Density (A/cm2)')  # Add a y-label to the axes.
        ax.grid()
        if self.Jphoto > 0.: #light JV
            ax.plot(self.Vpoints,self.Jpoints,\
                    marker='o',ls='',ms=12,c='#000000')  #special points
            #ax.scatter(self.Vpoints,self.Jpoints,s=100,c='#000000',marker='o')  #special points
            axr = ax.twinx()
            axr.plot(self.Vsingle, self.Psingle,ls='--',c='red')
            axr.set_ylabel('Power (W)')

            snote = 'Eg = {0:.2f} eV, Jpc = {1:.1f} mA/cm2, T = {2:.1f} C'\
                .format(self.Eg, self.Jphoto*1000, self.TC)
            snote += '\nGsh = {0:.1e} S, Rser = {1:g} Ω, A = {2:g} cm2 '\
                .format(self.Gsh, self.Rser, self.lightarea)
            snote += '\nVoc = {0:.2f} V, Jsc = {1:.1f} mA/cm2, FF = {2:.1f}%'\
                .format(self.Voc, self.Jsc*1000, self.FF*100)
            ax.text(-0.2,0,snote,bbox=dict(facecolor='white'))
            ax.set_title(title + " Light")  # Add a title to the axes.
        else:
            ax.set_yscale("log")
            ax.set_title(title + " Dark")  # Add a title to the axes.
    
        return fig