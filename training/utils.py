import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib as mpl
import mne
from mne.utils import _check_option, _validate_type, logger, _check_sphere
from mne.io import show_fiff, Info
from mne.io._digitization import _get_fid_coords
from mne.io.pick import (channel_type, channel_indices_by_type, pick_channels,
                       _pick_data_channels, _DATA_CH_TYPES_SPLIT,
                       _DATA_CH_TYPES_ORDER_DEFAULT, _VALID_CHANNEL_TYPES,
                       pick_info, _picks_by_type, pick_channels_cov,
                       _contains_ch_type)
from copy import deepcopy
from mne.io.constants import FIFF
from mne.transforms import apply_trans

from mne.defaults import _handle_default
from mne.viz.utils import _plot_sensors

def _init_config(data_path):
    config = {}
    config["n_class"] = 6
    config["n_sample"] = 32
    config["n_channel"] = 124

    config["sub_id"] = 1
    config["data_path"] = data_path
    config["add_self_loops"] = True
    config["edge_mode"] = "complete_graph"

    config["batch_size"] = 50
    config["num_epochs"] = 100
    config["lr"] = 0.001
    config["num_workers"] = 4
    config["weight_decay"] = 0.0001

    config['epochs'] = 100
    config["n_layers"] = 1
    config["input_dim"] = 32
    config["n_heads"] = 8
    config["dropout"] = 0.1
    config["edge_sample_ratio"] = 0.5

    return config


def init_model(exp_id, config=None):

    weight_path = "../../results/checkpoint"
    output_path = f"../../results/{exp_id}"
    data_path = "../../data/processed"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if config is None:
        config = _init_config(data_path)

    return config, weight_path, output_path, data_path





#---------------------------------------------------------------------------------

def plot2(ax, scale_factor=20, show_names=True, kind='topomap', show=True, sphere=None, verbose=None):

    m = mne.channels.make_standard_montage('GSN-HydroCel-128')
    return plot_montage(m, ax, scale_factor=scale_factor,
                        show_names=show_names, kind=kind, show=show,
                        sphere=sphere)


def plot_montage(montage, ax, scale_factor=20, show_names=True, kind='topomap', show=True, sphere=None, verbose=None):

    from scipy.spatial.distance import cdist
    from mne.channels import DigMontage, make_dig_montage
    from mne.io import RawArray
    from mne import create_info

    _check_option('kind', kind, ['topomap', '3d'])
    _validate_type(montage, DigMontage, item_name='montage')
    ch_names = montage.ch_names
    title = None

    if len(ch_names) == 0:
        raise RuntimeError('No valid channel positions found.')

    pos = np.array(list(montage._get_ch_pos().values()))

    dists = cdist(pos, pos)

    # only consider upper triangular part by setting the rest to np.nan
    dists[np.tril_indices(dists.shape[0])] = np.nan
    dupes = np.argwhere(np.isclose(dists, 0))
    if dupes.any():
        montage = deepcopy(montage)
        n_chans = pos.shape[0]
        n_dupes = dupes.shape[0]
        idx = np.setdiff1d(np.arange(len(pos)), dupes[:, 1]).tolist()
        logger.info("{} duplicate electrode labels found:".format(n_dupes))
        logger.info(", ".join([ch_names[d[0]] + "/" + ch_names[d[1]]
                               for d in dupes]))
        logger.info("Plotting {} unique labels.".format(n_chans - n_dupes))
        ch_names = [ch_names[i] for i in idx]
        ch_pos = dict(zip(ch_names, pos[idx, :]))
        # XXX: this might cause trouble if montage was originally in head
        fid, _ = _get_fid_coords(montage.dig)
        montage = make_dig_montage(ch_pos=ch_pos, **fid)

    info = create_info(ch_names, sfreq=256, ch_types="eeg")
    raw = RawArray(np.zeros((len(ch_names), 1)), info, copy=None)
    raw.set_montage(montage, on_missing='ignore')
    fig, pos = plot_sensors(info, axes=ax, kind=kind, show_names=show_names, show=show,
                       title=title, sphere=sphere)
    collection = fig.axes[0].collections[0]
    collection.set_sizes([scale_factor])

    # fig.suptitle(class_name, fontsize=30)

    return fig, pos

def plot_sensors(info, kind='topomap', ch_type=None, title=None,
                 show_names=False, ch_groups=None, to_sphere=True, axes=None,
                 block=False, show=True, sphere=None, pointsize=None,
                 linewidth=2, verbose=None):

    from mne.viz.evoked import _rgb

    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info not %s' % type(info))
    ch_indices = channel_indices_by_type(info)
    allowed_types = _DATA_CH_TYPES_SPLIT
    if ch_type is None:
        for this_type in allowed_types:
            if _contains_ch_type(info, this_type):
                ch_type = this_type
                break
        picks = ch_indices[ch_type]
    elif ch_type == 'all':
        picks = list()
        for this_type in allowed_types:
            picks += ch_indices[this_type]
    elif ch_type in allowed_types:
        picks = ch_indices[ch_type]
    else:
        raise ValueError("ch_type must be one of %s not %s!" % (allowed_types,
                                                                ch_type))

    if len(picks) == 0:
        raise ValueError('Could not find any channels of type %s.' % ch_type)

    # if not _check_ch_locs(info=info, picks=picks):
    #     raise RuntimeError('No valid channel positions found')

    dev_head_t = info['dev_head_t']
    chs = [info['chs'][pick] for pick in picks]
    pos = np.empty((len(chs), 3))
    for ci, ch in enumerate(chs):
        pos[ci] = ch['loc'][:3]
        if ch['coord_frame'] == FIFF.FIFFV_COORD_DEVICE:
            if dev_head_t is None:
                # warn('dev_head_t is None, transforming MEG sensors to head '
                #      'coordinate frame using identity transform')
                dev_head_t = np.eye(4)
            pos[ci] = apply_trans(dev_head_t, pos[ci])
    del dev_head_t

    ch_names = np.array([ch['ch_name'] for ch in chs])
    bads = [idx for idx, name in enumerate(ch_names) if name in info['bads']]
    if ch_groups is None:
        def_colors = _handle_default('color')
        colors = ['red' if i in bads else def_colors[channel_type(info, pick)]
                  for i, pick in enumerate(picks)]
    else:
        if ch_groups in ['position', 'selection']:
            # Avoid circular import
            from mne.channels import (read_vectorview_selection, _SELECTIONS,
                                    _EEG_SELECTIONS, _divide_to_regions)

            if ch_groups == 'position':
                ch_groups = _divide_to_regions(info, add_stim=False)
                ch_groups = list(ch_groups.values())
            else:
                ch_groups, color_vals = list(), list()
                for selection in _SELECTIONS + _EEG_SELECTIONS:
                    channels = pick_channels(
                        info['ch_names'],
                        read_vectorview_selection(selection, info=info))
                    ch_groups.append(channels)
            color_vals = np.ones((len(ch_groups), 4))
            for idx, ch_group in enumerate(ch_groups):
                color_picks = [np.where(picks == ch)[0][0] for ch in ch_group
                               if ch in picks]
                if len(color_picks) == 0:
                    continue
                x, y, z = pos[color_picks].T
                color = np.mean(_rgb(x, y, z), axis=0)
                color_vals[idx, :3] = color  # mean of spatial color
        else:
            import matplotlib.pyplot as plt
            colors = np.linspace(0, 1, len(ch_groups))
            color_vals = [plt.cm.jet(colors[i]) for i in range(len(ch_groups))]
        if not isinstance(ch_groups, (np.ndarray, list)):
            raise ValueError("ch_groups must be None, 'position', "
                             "'selection', or an array. Got %s." % ch_groups)
        colors = np.zeros((len(picks), 4))
        for pick_idx, pick in enumerate(picks):
            for ind, value in enumerate(ch_groups):
                if pick in value:
                    colors[pick_idx] = color_vals[ind]
                    break
    title = 'Sensor positions (%s)' % ch_type if title is None else title
    fig, pos = _plot_sensors(pos, info, picks, colors, bads, ch_names, title,
                        show_names, axes, show, kind, block,
                        to_sphere, sphere, pointsize=pointsize,
                        linewidth=linewidth)

    return fig, pos

def _plot_sensors(pos, info, picks, colors, bads, ch_names, title, show_names,
                  ax, show, kind, block, to_sphere, sphere, pointsize=None,
                  linewidth=2):
    """Plot sensors."""
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 analysis:ignore
    from mne.viz.topomap import _get_pos_outlines, _draw_outlines
    sphere = _check_sphere(sphere, info)

    edgecolors = np.repeat(rcParams['axes.edgecolor'], len(colors))
    edgecolors[bads] = 'red'
    axes_was_none = ax is None
    if axes_was_none:
        subplot_kw = dict()
        if kind == '3d':
            subplot_kw.update(projection='3d')
        fig, ax = plt.subplots(1, figsize=(10,10), subplot_kw=subplot_kw)
        # fig, ax = plt.subplots(1, figsize=(max(rcParams['figure.figsize']),) * 2, subplot_kw=subplot_kw)
    else:
        fig = ax.get_figure()

    if kind == '3d':
        print("...")
    else: 
        pointsize = 10 if pointsize is None else pointsize
        ax.text(0, 0, '', zorder=1)

        pos, outlines = _get_pos_outlines(info, picks, sphere,
                                          to_sphere=to_sphere)
        
        # print(pos)
        # print(pos.shape)
        global pos_2d
        pos_2d = pos

        _draw_outlines(ax, outlines)
        tmp = np.array([(x, y) for i,(x,y) in enumerate(pos) if i not in [47,118,124,125,126,127]])
        ref_dist, scale_factor = abs(pos[47,0] - pos[118,0]), 0.08
        pos[47,0] += ref_dist * scale_factor
        pos[118,0] -= ref_dist * scale_factor
        pts = ax.scatter(tmp[:, 0], tmp[:, 1], picker=True, clip_on=False,
                         c=colors[:122], edgecolors=edgecolors, s=pointsize,
                         lw=linewidth)
        # pts = ax.scatter(pos[:124, 0], pos[:124, 1], picker=True, clip_on=False,
        #                  c=colors[:124], edgecolors=edgecolors, s=pointsize,
        #                  lw=linewidth)
        
        fig.lasso = None

        # Equal aspect for 3D looks bad, so only use for 2D
        ax.set(aspect='equal')
        if axes_was_none:  # we'll show the plot title as the window title
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.axis("off")  # remove border around figure

        
        # ---------------------------------------------------------------------------------
        # ax.plot([pos[17][0], pos[15][0]], [pos[17][1], pos[15][1]])

        # cmap = mpl.cm.coolwarm
        # norm = mpl.colors.Normalize(vmin= 0, vmax=1)

        # for f, t, w in zip(edge_index[0], edge_index[1], edge_weight):
        #     x, y = [], []
        #     x.append(pos[f][0])
        #     x.append(pos[t][0])
        #     y.append(pos[f][1])
        #     y.append(pos[t][1])

        #     # project edge_weight from 0~1.0 to 0.1~5.0
        #     # w = w * 4.9 + 0.1
        #     if w < 0.35:
        #         continue
        #     w = (w - 0.3)*(2 - 0.3) + 0.3
        #     w = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba([w])
        #     ax.plot(x, y, linewidth=2, color=w[0])
            
        # # ---------------------------------------------------------------------------------

        # fig.savefig(f'{subject_id}_{class_name}.png', dpi=300)
    # .........
    return fig, pos

#---------------------------------------------------------------------------------



