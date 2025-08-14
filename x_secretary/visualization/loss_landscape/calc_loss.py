import torch,os
import numpy as np
from loguru import logger
from .directions import create_random_directions
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from pathlib import Path

def get_indices(vals, xcoordinates, ycoordinates):
    inds = np.array(range(vals.size)) 
    inds = inds[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]
    return inds, np.c_[s1, s2] 


def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0] # Direction vector present in the scale of weights
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)] #  αδ + βη
    
    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w + d # θ^* + αδ + βη


class LossLandscape():
    def __init__(self,
                 model,get_loss,
                 xmin=-1.0,xmax=1.0,xnum=51,
                 ymin=-1.0,ymax=1.0,ynum=51,
                 vmin=0.1,vmax=10,vlevel=0.2,
                ):
        '''
        Plot loss landscape

        ---

        Args:

            model: torch models for generating random weight directions

            get_loss: a function that return loss in each weight sample
                    should return {name, loss_value}

            xmin: Minimum value of x-coordinate

            xmax: Maximum value of x-coordinate

            xnum: Number of x-coordinate

            ymin: Minimum value of y-coordinate

            ymax: Maximum value of y-coordinate

            ynum: Number of y-coordinate

            vmin: Miminum value to map

            vmax: Maximum value to map

            vlevel: plot contours every vlevel

        '''
        self._args={
            'xmin': xmin, #'Minimum value of x-coordinate'),
            'xmax': xmax,  #'Maximum value of x-coordinate'),
            'xnum': xnum, #'Number of x-coordinate'),
            'ymin': ymin, #'Minimum value of y-coordinate'),
            'ymax': ymax,  #'Maximum value of y-coordinate'),
            'ynum': ynum, #'Number of y-coordinate'),
            'vmin': vmin, #'Miminum value to map'),
            'vmax': vmax, #'Maximum value to map'),
            'vlevel': vlevel,# 'plot contours every vlevel'),'save_imges', 'save as images'),
        }
        self._model=model
        self._get_loss=get_loss
        self._rslt_dict=None
        pass
    
    def update_plot_args(self,args: dict):
        self._args |= args

    @torch.no_grad()
    def calculate_loss_land_scape(self):
        rand_directions = create_random_directions(self._model)

        init_weights = [p.data for p in self._model.parameters()] # pretrained weights

        xcoordinates = torch.linspace(self._args['xmin'], self._args['xmax'], self._args['xnum'])
        ycoordinates = torch.linspace(self._args['ymin'], self._args['ymax'], self._args['ynum'])
        coords=torch.stack([*torch.meshgrid([ycoordinates,xcoordinates],indexing='ij')],dim=-1)

        rslt_dict={}

        for _i,(_y,_x) in enumerate(product(range(coords.shape[0]),range(coords.shape[1]))):
            # if _i>=100: break
            _coord=coords[_y,_x]

            overwrite_weights(self._model,init_weights,rand_directions,_coord)
            _results = self._get_loss()

            logger.info(f'Evaluating {_i}, coord={_coord}, results= {_results}')

            for _2_keys,_values in _results.items():
                if _2_keys not in rslt_dict:
                    rslt_dict[_2_keys]=torch.zeros(coords.shape[0],coords.shape[1])
                rslt_dict[_2_keys][_y,_x]=_values

        self._rslt_dict=rslt_dict
        return rslt_dict

    def visualize(self, loss_name,save_dir: str | Path,draw_heatmap=False):
        if self._rslt_dict is None:
            raise RuntimeError('No loss data have recorded yet!')

        result_file_path = os.path.join(save_dir, '2D_images/')
        if not os.path.isdir(result_file_path):
            os.makedirs(result_file_path)

        Z_LIMIT = 10
        xcoordinates = torch.linspace(self._args['xmin'], self._args['xmax'], self._args['xnum'])
        ycoordinates = torch.linspace(self._args['ymin'], self._args['ymax'], self._args['ynum'])
        Y,X=torch.meshgrid([ycoordinates,xcoordinates],indexing='ij')
        X=X.numpy()
        Y=Y.numpy()

        Z = self._rslt_dict[loss_name].numpy()

        #Z = np.log(Z)  # logscale

        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(Y, X, Z, cmap = 'summer', levels=np.arange(self._args["vmin"], self._args["vmax"], self._args["vlevel"]))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + loss_name + '_2dcontour' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        fig = plt.figure()
        CS = plt.contourf(Y, X, Z, levels=np.arange(self._args["vmin"], self._args["vmax"], self._args["vlevel"]))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + loss_name + '_2dcontourf' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        # Save 2D heatmaps image
        if draw_heatmap:
            import seaborn as sns
            plt.figure()
            sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=self._args["vmin"], vmax=self._args["vmax"],
                                   xticklabels=False, yticklabels=False)
            sns_plot.invert_yaxis()
            sns_plot.get_figure().savefig(result_file_path + loss_name + '_2dheat.pdf',
                                          dpi=300, bbox_inches='tight', format='pdf')

        # Save 3D surface image
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(Y, X, Z, linewidth=0, antialiased=True)
        fig.savefig(result_file_path + loss_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')