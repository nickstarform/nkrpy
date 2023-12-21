"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np

# relative modules

# global attributes
__all__ = ('SelectFromCollection', 'plotter')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1



class SelectFromCollection(object):
    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

class plotter(object):
    def __init__(self,title,size=[10,7]):
        self.size   = size
        self.title  = title
        self.data   = {}

    def close(self):
        plt.close(self.f[0])

    def open(self,numsubs=(1,1),xlabels=None,ylabels=None, xticks=None, yticks=None, xlabel=None, ylabel=None):
        self.numsubs = numsubs
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.xticks = xticks
        self.yticks = yticks
        self.f = plt.subplots(nrows=numsubs[0], ncols=numsubs[1],figsize=self.size)
        self.f[1].set_title(self.title, fontsize=30)
        if xlabels is not None and ylabels is not None:
            self.f[1].set_xlabel(xlabel)
            self.f[1].set_ylabel(ylabel)
            self.f[1].set_xticks(xticks)
            self.f[1].set_yticks(yticks)
            self.f[1].set_xticklabels(xlabels)
            self.f[1].set_yticklabels(ylabels)

    def refresh(self):
        self.f[1].set_xlabel(self.xlabel)
        self.f[1].set_ylabel(self.ylabel)
        self.f[1].set_xticks(self.xticks)
        self.f[1].set_yticks(self.yticks)
        self.f[1].set_xticklabels(self.xlabels)
        self.f[1].set_yticklabels(self.ylabels)

    def scatter(self,x,y,datalabel,**kwargs):
        self.data[datalabel] = self.f[1].scatter(x,y,**kwargs)

    def plot(self,x,y,datalabel,**kwargs):
        self.data[datalabel] = self.f[1].plot(x,y,**kwargs)

    def imshow(self,img, *args,**kwargs):
        self.f[1].imshow(img, *args,**kwargs)
        self.f[1].set_aspect(1./self.f[1].get_data_ratio())
        self.f[1].plot([0.5, 0.5], [0, 1], color='white', linewidth=3, linestyle='--', transform=self.f[1].transAxes, zorder=20)
        self.f[1].plot([0, 1], [0.5, 0.5], color='white', linewidth=3, linestyle='--', transform=self.f[1].transAxes, zorder=20)

    def int(self):
        plt.ion()

    def draw(self):
        #self.f[0].legend()
        self.f[0].canvas.draw_idle()

    def selection(self,label, prompt: str = None):
        temp      = []
        msk_array = []
        prompt = prompt if prompt is not None else 'total'
        prompt = f"Draw a mask around the {prompt} emission to fit the PV diagram to...."
        while True:
            selector = SelectFromCollection(self.f[1], self.data[label],0.1)
            print(prompt)
            self.draw()
            input('[RET] to accept selected points')
            temp = selector.xys[selector.ind]
            msk_array = np.append(msk_array,temp)
            selector.disconnect()
            # Block end of script so you can check that the lasso is disconnected.
            answer = input("(y or [SPACE]/n or [RET]) Want to draw another lasso region")
            self.f[0].show()
            if (answer.lower().startswith("n") or (answer == "")):
                self.save(f'{label}_PLOT.pdf')
                break
        answer = input("(y or [SPACE]/n or [RET]) Want to remove regions?")
        if not (answer.lower().startswith("n") or (answer == "")):
            while True:
                selector = SelectFromCollection(self.f[1], self.data[label], 0.1)
                print("Draw a mask around the max emission to remove....")
                self.draw()
                input('[RET] to accept selected points')
                temp = selector.xys[selector.ind]
                for t in temp:
                    if t in msk_array:
                        del msk_array[np.where(t == msk_array)]
                selector.disconnect()
                # Block end of script so you can check that the lasso is disconnected.
                answer = input("(y or [SPACE]/n or [RET]) Want to draw another lasso region")
                self.f[0].show()
                if ((answer.lower() == "n") or (answer == "")):
                    self.save(f'{label}_PLOT.pdf')
                    break
            

        return msk_array

    def set_xlim(self, *args):
        self.f[1].set_xlim(*args)

    def set_ylim(self, *args):
        self.f[1].set_ylim(*args)

    def save(self,name):
        plt.savefig(name)


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
