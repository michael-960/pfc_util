from .base import pfc_base, field as fd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np

_default_colors = ['blue', 'red', 'mediumseagreen', 'magenta', 'dodgerblue', 'limegreen', 'darkslategrey', 'orange']
_default_colors = ['steelblue', 'darkseagreen', 'palevioletred']

class PFC:
    def __init__(self, field: fd.RealField2D):
        matplotlib.use('TKAgg')
        matplotlib.style.use('fast')

        self.field = field 
        self.age = 0
        self.history = [] 
        self.fef = None
        self.current_minimizer = None

    def set_eps(self, eps):
        self.fef = pfc_base.PFCFreeEnergyFunctional(eps)

    def new_mu_minimizer(self, dt, eps, mu):
        self.fef = pfc_base.PFCFreeEnergyFunctional(eps)
        self.current_minimizer = pfc_base.ConstantChemicalPotentialMinimizer(self.field, dt, eps, mu)
        self.current_minimizer.set_age(self.age)

    def new_nonlocal_minimizer(self, dt, eps):
        self.fef = pfc_base.PFCFreeEnergyFunctional(eps)
        self.current_minimizer = pfc_base.NonlocalConservedMinimizer(self.field, dt, eps)
        self.current_minimizer.set_age(self.age)

    def evolve(self, N_steps, N_epochs):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 

        self.current_minimizer.run_multisteps(N_steps, N_epochs)
        self.history.append(self.current_minimizer.history)
        self.age = self.current_minimizer.age

    def evolve_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 
        self.current_minimizer.run_nonstop(N_steps, custom_keyboard_interrupt_handler)
        self.history.append(self.current_minimizer.history)
        self.age = self.current_minimizer.age

    def field_snapshot(self):
        return self.field.export_state()     

    def plot_history(self, *item_names, colors=_default_colors, show=True):

        nrows = len(item_names)
        #fig, axs = plt.subplots(nrows=nrows, ncols=1, gridspec_kw={'height_ratios': [0.5,0.5] + [10]*(nrows-3) + [1]})

        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.08, right=0.92)
        outer = gridspec.GridSpec(3, 1, height_ratios=[1.5,32,2], hspace=0.1) 
        gs_sliders = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1,1], subplot_spec=outer[0], hspace=0.2)
        gs_plots = gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec = outer[1], hspace=0)
        gs_time = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2], hspace=0)

        axslider1 = plt.subplot(gs_sliders[0])
        axslider1.xaxis.tick_top()
        axslider2 = plt.subplot(gs_sliders[1])
        axs = [plt.subplot(cell) for cell in gs_plots]
        axT = plt.subplot(gs_time[0])

        for i, item_name in enumerate(item_names):
            ax = axs[i]
            ax.tick_params(axis='x', direction='in', pad=-12)
            ax.tick_params(axis='y', direction='in', pad=-13)

            for tick in ax.get_yticklabels():
                tick.set_ha('left')

            for j, block  in enumerate(self.history):
                ax.plot(block.get_item('t'), block.get_item(item_name), color=colors[j%len(colors)], lw=1)
                ax.get_xticklabels()[0].set_ha('left')
                ax.set_ylabel(block.get_item_latex(item_name))

        age_i = 0
        time_boxes = []
        time_coords = []
        for j, block in enumerate(self.history):
            age_f = block.age
            if j != 0:
                age_i = self.history[j-1].age

            for i, item_name in enumerate(item_names):
                ax = axs[i] 
                if block.get_item(item_name)[0] is None:
                    ax.add_patch(Rectangle((age_i, -1e6), age_f-age_i, 2e6, facecolor='grey', edgecolor='none'))
 
            rx, ry, rw, rh = age_i, 0, age_f-age_i, 1
            cx, cy = rx+rw/2, ry+rh/2
            time_coords.append((rx, ry, rw, rh))
            #rect = Rectangle((rx, ry), rw, rh, facecolor='oldlace', edgecolor='k')
            rect = Rectangle((rx, ry), rw, rh, facecolor=colors[j%len(colors)], edgecolor='none')
            time_boxes.append(rect)
            axT.add_artist(rect)
            #axT.annotate(block.label, (cx, cy), color=colors[j%len(colors)], ha='center', va='center', fontsize=10, clip_on=True)
            annot_alt = block.label.replace(' ', '\n', 1)
            axT.annotate(annot_alt, (cx, cy), color='oldlace', ha='center', va='center', fontsize=10, clip_on=True)
            #axT.text((age_i+age_f)/2, 0.5, block.minimizer.label, color=colors[j%len(colors)], ha='center', va='center',
            #        fontsize=10, clip_on=True)

        axT.set_yticks([])
        axT.set_xlabel('$t$', fontsize=12)
       
        
        width = 40
        slider1 = Slider(axslider1, 't', 0, max(self.age-width, 0), valinit=0, valfmt='%.3f', color='mediumorchid')
        slider2 = Slider(axslider2, '$\Delta t$', 1, self.age-1, valinit=self.age-1, valfmt='%.3f', color='plum')
        axslider2.set_xticks([1, self.age/2, self.age-1])

        axslider1.add_artist(axslider1.xaxis)
        axslider2.add_artist(axslider2.xaxis)

        def update1(val):
            for i in range(nrows):
                axs[i].set_xlim(slider1.val, slider1.val+slider2.val)
            axT.set_xlim(slider1.val, slider1.val+slider2.val)

        def update2(val):
            for i in range(nrows):
                axs[i].set_xlim(slider1.val, slider1.val+slider2.val)
            axT.set_xlim(slider1.val, slider1.val+slider2.val)

            slider1_max = max(self.age-slider2.val, 0) 
            slider1_new_val = min(slider1.val, slider1_max)
            slider1.set_val(slider1_new_val)
            slider1.valmax = slider1_max
            axslider1.set_xlim(0, slider1_max)
            axslider1.set_xticks([0, slider1_max/4, slider1_max/2, slider1_max*3/4, slider1_max])

        slider1.on_changed(update1)
        slider2.on_changed(update2)

        update1(0)
        update2(width)

        # tooltip
        annot = axT.annotate("", xy=(0,0), xytext=(0,0), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"), ha='center')
        annot.set_visible(False)

        fig = plt.gcf()
        def update_annot(i, xpos):
            block = self.history[i]
            annot.set_text(block.label) 
            annot.xy = (xpos, time_coords[i][1] + time_coords[i][3])
            
            annot.get_bbox_patch().set_alpha(0.9)

        def on_hover(event):
            vis = annot.get_visible()
            if event.inaxes == axT:
                for i, time_box in enumerate(time_boxes):
                    cont = time_box.contains(event)
                    if cont[0]:
                        xpos = event.xdata
                        update_annot(i, xpos)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()



        fig.canvas.mpl_connect('motion_notify_event', on_hover)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        widgets = [slider1, slider2]


        if show:
            plt.show()
        else:
            return axT, axs, axslider1, axslider2, widgets 

    def save(self, path):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()


def load_model(path):
    saved = np.load(path, allow_pickle=True)
    raise NotImplementedError()

class PFCHistory:
    def __init__(self):
        self.content = []

    
