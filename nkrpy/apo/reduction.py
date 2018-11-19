"""Provides common functions for reducing TSPEC data."""

# fit continuum
# fit line
# eqwidth
# pretty plotting


# make class per target
# allow for a lot of data to be loaded
# save to output file for quick load without loading fits again


def plotting(ax,xmin,xmax,x,y,tempsource,line,count,start=False):
    colours = ['orange','black','blue','red',\
              'green','grey','purple']
    colour = colours[count%len(colours)]
    y = np.array(y)
    x = np.array(x)
    origx = x.copy()
    origy = y.copy()
    x = x[~np.isnan(origy)]
    y = y[~np.isnan(origy)]
     
    print("Count: {},Before: {},{}".format(count,x.shape,y.shape))
    if start == False:
        temp = []
        if count == 0:
            for i,j in enumerate(x):
                if (j < 1.7):
                    temp.append(i)
        elif count == 1:
            for i,j in enumerate(x):
                if ((1.75 < j) or (j < 1.5)):
                    temp.append(i)
        elif count == 2:
            for i,j in enumerate(x):
                if (1.33 < j) or (j < 1.17):
                    temp.append(i)
        elif count == 3:
            for i,j in enumerate(x):
                if ((j < 1.05) or ((1.11<j)) and (j<1.17)): 
                    temp.append(i)
        elif count == 4:
            for i,j in enumerate(x):
                if (j < 0.95):
                    temp.append(i)
        temp = np.array(temp)
        temp.sort()
        tempx = np.delete(x,temp)
        tempy = np.delete(y,temp)
        expected = [1.,1.]
        params,cov = curve_fit(linear,tempx,tempy,expected)
        #print(params)
        if len(temp) > 0:
            y[temp[0]] = linear(x[temp[0]],*params)
            y[temp[len(temp)-1]] = linear(x[temp[len(temp)-1]],*params)
            x = np.delete(x,temp[1:len(temp)-1])
            y = np.delete(y,temp[1:len(temp)-1])

        print("After: {},{}".format(x.shape,y.shape))
        if x.shape[0] == 0:
            x = origx
            y = origy
        count +=1
    ax.plot(x,y,'-',color=colour,label=tempsource[-6:])
    for f in line:
        #print(f)
        if f == 'brg':
            naming = r'Br $\gamma$'
        elif f == 'pab':
            naming = r'Pa $\beta$'
        elif f == 'pag':
            naming = r'Pa $\gamma$'
        else:
            naming = f
        for pl,pj in enumerate(line[f]):
            #print(pl,pj)
            if (pl < (len(line[f]))):
                if pj > 100:
                    pj = pj/10000
                val = int(int(min(range(len(x)),key=lambda i:abs(x[i]-pj))))
                if (xmin <= pj <= xmax )and (min(x) <= pj <= max(x) ):
                    if 10<=val<len(x)-11:
                        region=y[val-10:val+10] 
                    elif val > 0:
                        region=y[val:val+10]
                    elif val<len(x)-1:
                        region=y[val-10:val]
                    else:
                        region = y[val]
                    try:
                        linepos = max(region)
                    except ValueError:
                        linepos = 5*np.nanmean(x)
                    ax.text(pj, linepos*1.05, naming,
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        fontsize=10, color='red',rotation='vertical')

                    ax.plot((pj,pj),(linepos,linepos*1.05),'r-')

# end of file
