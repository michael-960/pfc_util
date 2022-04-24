from torusgrid import field_util
from torusgrid import fields as fd

from torusgrid.fields import RealField2D
from .core.base import PFCStateFunction, get_latex
from .core.evolution import PFCMinimizerHistory, import_minimizer_history

from michael960lib.common import IllegalActionError, with_type

from typing import List, Dict

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

import numpy as np


_default_colors = ['steelblue', 'darkseagreen', 'palevioletred', 'burlywood']


class PFCHistoryBlock:
    def __init__(self):
        raise NotImplementedError()

    def export_state(self):
        raise NotImplementedError()

    def import_state(self):
        raise NotImplementedError()

    def get_label(self) -> str:
        raise NotImplementedError()

    def has_span(self) -> bool:
        raise NotImplementedError()

    def get_final_field_state(self):
        raise NotImplementedError()

    def get_age_span(self) -> float:   
        raise NotImplementedError()

    def get_state_functions(self)-> Dict[str, List[float]]:
        raise NotImplementedError()

    def get_t(self) -> List[float]:
        raise NotImplementedError()

    def export(self):
        raise NotImplementedError()

class PFCInitialHistoryBlock(PFCHistoryBlock):
    def __init__(self, field: RealField2D, label):
        self.label = label
        self.field = field

    def get_label(self):
        return self.label

    def has_span(self):
        return False

    def get_age_span(self):
        return 0 

    def get_final_field_state(self):
        return self.field.export_state()

    def get_state_functions(self):
        raise IllegalActionError('there is no state functions associated with an edit action')

    def get_t(self):
        raise IllegalActionError('there is no time associated with an edit action')

    @with_type('initial')
    def export(self):
        return {
            'label': self.label,
            'field': self.field.export_state()
        }


class PFCMinimizerHistoryBlock(PFCHistoryBlock):
    def __init__(self, minimizer_history: PFCMinimizerHistory):
        if not minimizer_history.is_committed():
            raise IllegalActionError('cannot initialize history block with uncommitted minimizer history')
        self.minimizer_history = minimizer_history

        self.state_functions = {
            'Lx': [],
            'Ly': [],
            'f': [],
            'F': [],
            'psibar': [],
            'omega': [],
            'Omega': [],
        }

        keys = self.state_functions.keys()

        for sf in self.minimizer_history.get_state_functions():
            for item_name in keys:
                self.state_functions[item_name].append(sf.get_item(item_name))

        self.t = np.array(self.minimizer_history.t)
        self.label = minimizer_history.get_label()

    def get_label(self):
        return self.label

    def get_final_field_state(self):
        return self.minimizer_history.get_final_field_state()

    def has_span(self):
        return True

    def get_age_span(self):
        return self.minimizer_history.age
    
    def get_state_functions(self):
        return self.state_functions
    
    def get_t(self):
        return self.t

    @with_type('minimizer')
    def export(self) -> dict:
        state = {'minimizer_history': self.minimizer_history.export(), 'label': self.label}
        return state


class PFCEditActionHistoryBlock(PFCHistoryBlock):
    def __init__(self):
        pass

    def get_label(self):
        pass

    def has_span(self):
        return False

    def get_age_span(self):
        return 0 

    def get_final_field_state(self):
        pass

    def get_state_functions(self):
        raise IllegalActionError('there is no state functions associated with an edit action')

    def get_t(self):
        raise IllegalActionError('there is no time associated with an edit action')

    @with_type('edit_action')
    def export(self) -> dict:
        raise NotImplementedError()


def import_initial_history_block(state: dict) -> PFCInitialHistoryBlock:
    block = PFCInitialHistoryBlock(fd.import_field(state['field']), state['label'])
    return block


def import_minimizer_history_block(state: dict) -> PFCMinimizerHistoryBlock:
    minimizer_history = import_minimizer_history(state['minimizer_history'])
    block = PFCMinimizerHistoryBlock(minimizer_history)
    return block


def import_edit_action_history_block(state: dict) -> PFCEditActionHistoryBlock:
    raise NotImplementedError()


def import_history_block(state: dict) -> PFCHistoryBlock:
    if state['type'] == 'minimizer':   
        return import_minimizer_history_block(state)
    if state['type'] == 'initial': 
        return import_initial_history_block(state)
    if state['type'] == 'edit_action':   
        return import_edit_action_history_block(state)

    raise ValueError(f'{state["type"]} is not a valid history block type')



class PFCHistory:
    def __init__(self, field: RealField2D):
        self.blocks = []
        self.age_record = []
        self.append(PFCInitialHistoryBlock(field, 'loaded profile'))

    def append(self, block: PFCHistoryBlock):
        if not isinstance(block, PFCHistoryBlock):
            raise ValueError('only PFCHistoryBlock objects can be appended into a PFCHistory')
        self.blocks.append(block)
        max_age = self.get_max_age()

        if block.has_span():
            self.age_record.append(max_age + block.get_age_span())
        else:
            self.age_record.append(max_age)

    def cut_and_insert(self, block: PFCHistoryBlock, index):
        self.blocks = self.blocks[:index]
        self.append(block)
    
    def get_max_age(self):
        max_age = 0
        if len(self.age_record) > 0:
            max_age = self.age_record[-1]
        return max_age 

    def get_blocks(self) -> List[PFCHistoryBlock]:
        return self.blocks.copy()

    def get_block(self, i) -> PFCHistoryBlock:
        return self.blocks[i]

    def get_block_age_lim(self, i):
        if i == 0:
            return 0, 0
        return self.age_record[i-1], self.age_record[i]

    # initial field state of blocks[i]
    def get_field_state(self, i):
        if i == 0:
            return
        return self.blocks[i-1].get_final_field_state()

    def plot(self, *item_names, colors=_default_colors, start=0, end=-1, show=True):
        nrows = len(item_names)
        if nrows == 0:
            raise ValueError('cannot plot history with zero items')

        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.08, right=0.92)
        outer = gridspec.GridSpec(3, 1, height_ratios=[1.5,32,2], hspace=0.1) 
        gs_sliders = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1,1], subplot_spec=outer[0], hspace=0.2)
        gs_plots = gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec = outer[1], hspace=0.1)
        gs_time = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2], hspace=0)

        axslider1 = plt.subplot(gs_sliders[0])
        axslider1.xaxis.tick_top()
        axslider2 = plt.subplot(gs_sliders[1])
        axs = [plt.subplot(cell) for cell in gs_plots]
        axT = plt.subplot(gs_time[0])

        time_coords = []
        time_boxes = []

        if end < 0:
            end = len(self.get_blocks()) + end


        color_index = 0
        for j, block in enumerate(self.get_blocks()):
            if j < start or j > end:
                continue
            age_span = block.get_age_span()
            age_i, age_f = self.get_block_age_lim(j)
            rx, ry, rw, rh = age_i, 0, age_f-age_i, 1
            cx, cy = rx+rw/2, ry+rh/2


            rect = Rectangle((rx, ry), rw, rh, facecolor=colors[color_index%len(colors)], edgecolor='none')
            time_coords.append((rx, ry, rw, rh))
            time_boxes.append(rect)

            if block.has_span():
                sfs = block.get_state_functions()
                t = block.get_t() 
                

                axT.add_artist(rect)
                annot_alt = block.get_label().replace(' ', '\n', 1)
                axT.annotate(annot_alt, (cx, cy), color='oldlace', ha='center', va='center', fontsize=10, clip_on=True)


                for i, item_name in enumerate(item_names):
                    ax = axs[i]
                    ax.plot(t+age_i, sfs[item_name], color=colors[color_index%len(colors)], lw=1)
                    ax.set_ylabel(get_latex(item_name))
                    
                    ax.plot([age_i, age_i], [-1e2, 1e2], lw=1, linestyle=':', color='k')

                    if sfs[item_name][0] is None:
                        ax.add_patch(Rectangle((age_i, -1e2), age_f-age_i, 2e2, facecolor='grey', edgecolor='none'))

                color_index += 1

            else:
                pass

        for i, item_name in enumerate(item_names):
            #axs[i].tick_params(axis='x', direction='in', pad=-12)
            #axs[i].tick_params(axis='y', direction='in', pad=-13)
            #axs[i].get_xticklabels()[0].set_ha('left')
            
            ymin = 1e4
            ymax = -1e4
            for j, block in enumerate(self.get_blocks()): 
                if block.has_span():
                    ydata = block.get_state_functions()[item_name]
                    if None in ydata:
                        continue
                    ymin = min(np.min(ydata), ymin)
                    ymax = max(np.max(ydata), ymax)
            
            ydelta = ymax - ymin
            axs[i].set_ylim(ymin - ydelta*0.05, ymax + ydelta*0.05)

            for tick in axs[i].get_yticklabels():
                pass
                #tick.set_ha('left')


        axT.set_yticks([])
        axT.set_xlabel('$t$', fontsize=12)
        
        width = 40

        age_start = self.get_block_age_lim(start)[0]
        age_end = self.get_block_age_lim(end)[1]
        age_range = age_end - age_start
        

        max_width = 150
        zoom_i = 1e-6
        zoom_f = 100-1e-6
        slider2_init = min(max(zoom_i, 100 * (1-max_width/age_range)), zoom_f)

        width = age_range * (100-slider2_init) / 100

        slider1 = Slider(axslider1, '$t$', age_start, age_start+width, valinit=age_start, valfmt='%.3f', color='mediumorchid')
        slider2 = Slider(axslider2, 'zoom', zoom_i, zoom_f, valinit=slider2_init, valfmt=f'%.3f%%', color='plum')

        axslider2.set_xticks([zoom_i, zoom_f], ['-', '+'])

        axslider1.add_artist(axslider1.xaxis)
        axslider2.add_artist(axslider2.xaxis)

        def update1(val):
            width = age_range * (100-slider2.val) / 100
            for i in range(nrows):
                axs[i].set_xlim(slider1.val, slider1.val+width)
            axT.set_xlim(slider1.val, slider1.val + width)

        def update2(val):
            width = age_range * (100-slider2.val) / 100
            for i in range(nrows):
                axs[i].set_xlim(slider1.val, slider1.val+width)
            axT.set_xlim(slider1.val, slider1.val + width)

            slider1_max = max(age_end-width, age_start) 

            slider1_new_val = min(slider1.val, slider1_max)

            slider1.set_val(slider1_new_val)
            slider1.valmax = slider1_max
            axslider1.set_xlim(age_start, slider1_max)

            tmp = slider1_max - age_start
            axslider1.set_xticks([age_start, age_start + tmp/4, age_start + tmp/2, age_start + tmp*3/4, age_start + tmp])

        slider1.on_changed(update1)
        slider2.on_changed(update2)

        update1(age_start)
        update2(slider2_init)

        # tooltip
        annot = axT.annotate("", xy=(0,0), xytext=(0,0), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"), ha='center')
        annot.set_visible(False)

        fig = plt.gcf()
        def update_annot(i, xpos):
            block = self.get_block(i)
            annot.set_text(block.get_label()) 
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

    def export(self):
        return {
            'blocks': [block.export() for block in self.blocks],
            'age_record': self.age_record,
        }


def import_history(state: dict) -> PFCHistory:
    initial_field = import_initial_history_block(state['blocks'][0]).field
    pfc_history = PFCHistory(initial_field)

    pfc_history.blocks = [import_history_block(block_state) for block_state in state['blocks']]
    pfc_history.age_record = state['age_record']

    return pfc_history

