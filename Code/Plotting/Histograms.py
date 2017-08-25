from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import Grid
import numpy as np
import Code.commons as cms
import Code.Preprocessing.MakeDatabase as Db


# Slice eyedata into an array containing information per trial, using two image events as slice indexes.
# TODO: This function needs to be tested AND FINISHED (How to save imageid from taskfile if only returns eyefile?)
def slice_file(_fileeye, _filetask, _events=None, _delimiters=None):
    if _events is None:
        _events = ['main_image_start', 'main_image_end']
    if _delimiters is None:
        _delimiters = ['\t', ',']

    # First load subject data
    eyedata = Db.file2data(_fileeye, _delimiters[0])
    taskdata = Db.file2data(_filetask, _delimiters[1])

    # Get indexes where _events happen
    event_on_times = []
    event_off_times = []
    for index in range(len(taskdata)):
        a_task_line = taskdata[index]
        # Get current task on that line and it's eventid dictionary
        current_task = int(a_task_line[1])
        eventsid = cms.eventsid[current_task]
        event_code_on = eventsid[_events[0]]
        event_code_off = eventsid[_events[1]]
        # Save times which an event occurs
        if float(a_task_line[5]) == event_code_on:
            event_on_times.append(a_task_line)
        if float(a_task_line[5]) == event_code_off:
            event_off_times.append(a_task_line)
    return eyedata
    # Make cuts and append array according to quantity of data


# Plots values over time and also makes horizontal and vertical histograms of count or probability
def plot_timehist2d(_fig, _time, _data, _bins, _lim, _normed=True, _label=('', '')):
    # Generate grid to plot: mainaxis has hist2d plot and mean curve.
    mygrid = gridspec.GridSpec(3, 3)
    ax_main = Grid(_fig, mygrid[1:3, 0:2], (1, 1))[0]
    ax_timehist = Grid(_fig, mygrid[0:1, 0:2], (1, 1))[0]
    ax_hist = Grid(_fig, mygrid[1:3, 2:3], (1, 1))[0]
    ax_main.get_shared_x_axes().join(ax_main, ax_timehist)
    ax_main.get_shared_y_axes().join(ax_main, ax_hist)

    # Calculate bin center values for mean curve
    binedges = np.linspace(*_lim[0], num=11)
    bincenters = [(binedges[i] + binedges[i + 1]) / 2 for i in range(len(binedges) - 1)]
    x_means = []
    x_var = []  # We should use std not sem value
    # For each bin we calculate mean and deviation values
    for t_ini, t_fin in zip(binedges[:-1], binedges[1:]):
        mask = (t_ini <= _time) & (_time < t_fin)  # type: np.ndarray
        x_means.append(np.mean(_data[mask]))
        x_var.append(np.std(_data[mask]))

    # Add main 2d-histogram figure with mean curve
    ax_main.hist2d(_time, _data, _bins, _lim, cmap='binary')
    ax_main.errorbar(bincenters, x_means, yerr=x_var, color='red')
    ax_main.grid()
    ax_main.set_xlim(_lim[0])
    ax_main.set_ylim(_lim[1])
    ax_main.set_xlabel(_label[0])
    ax_main.set_ylabel(_label[1])

    # Add two histograms of axis
    if _normed:
        hist_label = 'Probability'
    else:
        hist_label = 'Count'
    ax_timehist.hist(_time, bins=_bins[0], range=_lim[0], normed=_normed, color='black')
    ax_timehist.grid()
    ax_timehist.set_ylabel(hist_label)

    ax_hist.hist(_data, bins=_bins[1], range=_lim[1], normed=_normed, orientation='horizontal', color='black')
    ax_hist.grid()
    ax_hist.set_xlabel(hist_label)

# MAIN ROUTINE: Make histograms
if __name__ == '__main__':
    # First, we'll replicate time histogram figure from junji's for one session
    # We'll Make a database list both for Osaka and Chile subject
    osakaSubject = Db.OsakaSubject()
    _, filetask_osaka = osakaSubject.makedatabaselist('_task')
    _, fileeye_osaka = osakaSubject.makedatabaselist('_eye')
    chileSubject = Db.ChileSubject()
    _, filetask_chile = chileSubject.makedatabaselist('_task')
    _, fileeye_chile = chileSubject.makedatabaselist('_eye')

    taskfiles = np.append(filetask_osaka, filetask_chile)
    eyefiles = np.append(fileeye_osaka, fileeye_chile)

    # Slice each file per trials
    for taskfile, eyefile in zip(taskfiles, eyefiles):
        slices = slice_file(eyefile, taskfile)

    # Concatenate slices per subject
    test = 0
# TODO: Finish main routine !
# TODO: For each set of files do time-based 2D histograms.
