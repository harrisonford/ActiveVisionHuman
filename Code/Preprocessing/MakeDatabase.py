import re
import scipy.io as spio
from cili.util import *
from Code import commons as cms


# This function loads a data file into a tuple of values for each line, splitted by a delimiter
def file2data(_filepath, _delimiter, _noheader=True):
    # Read data lines
    with open(_filepath) as f:
        data_lines = f.readlines()

    # Iterate over each line and split by delimiter
    data = []
    for line in data_lines:
        data.append(line.split(_delimiter))

    header_indexes = []
    current_index = 0
    if _noheader:  # Then we eliminate header: if first item in each line is string we remove it from data until false.
        for a_line in data:
            # Map to string the first element of the line
            try:
                float(a_line[0])  # If first element is a number string then this can be done, therefore no more header
                break
            except ValueError:  # If string couldn't be handled as a number then it's header so we save index
                header_indexes.append(current_index)
                current_index += 1
        for index in sorted(header_indexes, reverse=True):  # We pop header from last to first line
            data.pop(index)
    return data


class SubjectType(object):
    def __init__(self, _filein=None):
        if _filein is not None:
            self.filepath = _filein
            self.name = self.filepath2name()
            self.recording, self.block, self.taskname = self.name2info()  # Adds recording, block and taskname
            self.samps, self.events = load_eyelink_dataset(_filein)
        else:
            self.filepath = None

    # This loader function must be overriden in each subclass
    def load(self, _filein=None):
        return SubjectType(_filein)

    # From the full path name we take only the subjects name code: ".../name.asc" = name
    def filepath2name(self):
        subtext = re.search('/([0-9A-Z]+?).asc', self.filepath)
        if subtext:
            return subtext.group(1)
        else:
            return None

    # This function is needed for some of Junji's work: each file needs to have session, recording and block number
    # In a generic way each recording is the first, there's only one block with no task given = Global
    def name2info(self):
        return 1, 1, 'Global'

    # A function to set subject's task, it can even change during experiment, check taskid dictionary in commons
    def task(self, _taskname):
        self.taskname = _taskname

    # Just dummy functions, needs to be made on each class: eyeevents and imageevents2dat depends on experiment
    def eyeevents2dat(self, _fileout):
        raise ValueError('SubjectClass root eyeevents2dat function should never be called, please override it.')

    def imageevents2dat(self, _fileout):
        raise ValueError('SubjectClass root imageevents2dat function should never be called, please override it')

    # get blink data from a subject with loaded eye events and samples
    def getblinks(self):
        if self.filepath is None:
            raise ValueError("Tried to get blinks from a subject with no filepath")
        _blinkdata = []
        for i_blink, _blink in self.events.dframes['EBLINK'].iterrows():
            idx_on = i_blink
            idx_off = i_blink + _blink.duration
            _blinkdata.append((idx_on, idx_off))
        return np.array(_blinkdata, dtype=[('on', long), ('off', long)])

    # getsaccades may use blinkdata to clean anidated saccades by a blink, which is a common noise in eyelink data
    def getsaccades(self, blinkdata=None):
        if self.filepath is None:
            raise ValueError("Tried to get saccades from a subject with no filepath")
        _saccdata = []
        for i_sac, sac in self.events.dframes['ESACC'].iterrows():
            idx_on = i_sac
            idx_off = i_sac + sac.duration
            # If blink data given reject saccades that contain a blink
            if blinkdata is not None:
                _mask = (idx_on <= blinkdata['on']) & (blinkdata['off'] <= idx_off)  # type: np.ndarray
            else:
                _mask = [0] * len(idx_on)
            if np.any(_mask):
                continue
            # EYELINK seems to erroneously register a saccade at the beginning of
            # recording. Such saccades have almost zero amplitude and duration of
            # one sample. They are rejected here.
            if sac.x_start - sac.x_end < 10 and sac.y_start - sac.y_end < 10 and sac.duration <= 4:
                continue
            amp = np.hypot(sac.x_end - sac.x_start, sac.y_end - sac.y_start)
            _saccdata.append((idx_on, idx_off, sac.x_start, sac.y_start, sac.x_end, sac.y_end, sac.peak_velocity, amp))
        return np.array(_saccdata,
                        dtype=[('on', long), ('off', long), ('x_on', float), ('y_on', float), ('x_off', float),
                               ('y_off', float), ('param1', float), ('param2', float)])

    # get fix data from a subject with loaded eye events and samples
    def getfixations(self):
        if self.filepath is None:
            raise ValueError("Tried to get fixations from a subject with no filepath")
        fixdata = []
        for i_fix, fix in self.events.dframes['EFIX'].iterrows():
            idx_on = i_fix
            idx_off = i_fix + fix.duration
            fixdata.append((idx_on, idx_off, fix.x_pos, fix.y_pos, 0, 0, fix.duration, 0))
        return np.array(fixdata,
                        dtype=[('on', long), ('off', long), ('x_on', float), ('y_on', float), ('x_off', float),
                               ('y_off', float), ('param1', float), ('param2', float)])

    # A function that just runs many eyeevents at once
    def eyeevents2dats(self, _filesin, _filesout):
        for _i in range(len(_filesin)):
            print "\tReading: ", _filesin[_i]
            _subject = self.load(_filesin[_i])
            print "\tWriting: ", _filesout[_i]
            _subject.eyeevents2dat(_filesout[_i])

    # A function that runs imageevents for many subjects
    def imageevents2dats(self, _filesin, _filesout):
        for _i in range(len(_filesin)):
            print "\tReading: ", _filesin[_i]
            _subject = self.load(_filesin[_i])
            print "\tWriting: ", filesout[_i]
            _subject.imageevents2dat(_filesout[_i])

    # Change (0,0) of X,Y representation to image center
    def centercoordinates(self, _datax=None, _datay=None):
        screen_size = self.getscreensize()
        datax = None
        datay = None
        if _datax is not None:
            datax = _datax - screen_size[0] / 2
        if _datay is not None:
            datay = _datay - screen_size[1] / 2
        return datax, datay

    def getscreensize(self):
        gaze_coords = None
        for i_msg, msg in self.events.dframes['MSG'].iterrows():
            if msg.label == 'GAZE_COORDS':
                gaze_coords = map(float, msg.content.split())
                break
        if gaze_coords is None:
            raise ValueError("centercoordinates: Could not find gaze coordinates in ascii file in" + self.filepath)
        screen_size = (gaze_coords[2] - gaze_coords[0], gaze_coords[3] - gaze_coords[1])
        return screen_size


class OsakaSubject(SubjectType):
    # On Osaka notation each subject is named by the task using F, S, M, etc.
    task_dictionary = dict(F='Free Viewing', S='Search', M='Memory')

    # Osaka notation of tags in asc file and it's conversion to OFFICIAL event tags (check commons)
    tag_dictionary = dict(FIXONNN='trial_start', FIXOFFF='fix_point_off', OBJONNN='object_image_start',
                          OBJOFFFF='object_image_end', SYNCTIME='main_image_start', ENDTIME='main_image_end',
                          TRIAL_END='trial_end')

    def load(self, _filein=None):
        return OsakaSubject(_filein)

    def name2info(self):
        # Osaka data structure: {sbj}{rec}00{blk}{tsk: symbol}
        recording = re.search('([1-9]+?)00', self.name).group(1)
        block = re.search('00([1-9]+?)', self.name).group(1)
        subject_task = re.search('([0-9]+)([A-Z]?)', self.name).group(2)
        taskname = self.task_dictionary[subject_task]
        return recording, block, taskname

    def imageevents2dat(self, _fileout, _delimiter=','):
        header = cms.imageheader
        msgs = self.events.dframes['MSG']
        task_id = cms.tasksid[self.taskname]
        event_dictionary = cms.imagesid[cms.tasksid[self.taskname]]

        # Center and convert to degrees X,Y samples
        self.samps.x_l, self.samps.y_l = self.centercoordinates(self.samps.x_l, self.samps.y_l)
        self.samps.x_l, self.samps.y_l = self.pixels2degrees(self.samps.x_l, self.samps.y_l)

        # Load .mat file containing information about trial success and image presented
        matpath = self.filepath[:-3] + 'mat'  # Same filepath and name but with 'mat' at the end instead
        mat_infos = spio.whosmat(matpath)
        mat_names, mat_sizes, mat_types = zip(*mat_infos)
        mat_data = spio.loadmat(matpath, variable_names=list(mat_names[:-1]))
        response = np.array(mat_data['dataSubjResp'])  # Variable contains: image_num | response | correct res | NaN
        image_ids = np.array(mat_data['dataStimConfig'])  # Variable contains: 'name.png'

        # We will write image events as lines info into a .dat file
        image_lines = [_delimiter.join(header)]
        image_lines.insert(0, 'task')  # This is just a format thing: first line must say "task"

        # For each set of start-end we check which lines to append
        # Slice indexes per trials: Fix onset -> image/object -> object/image -> endtime. Depending on task.
        indexes_trial_on = np.where(msgs == 'FIXONNN')[0]
        indexes_trial_off = np.where(msgs == 'TRIAL_END')[0]
        for trial in range(len(response)):
            trial_success = int(response[trial, 1] == response[trial, 2])  # Response matches right answer
            image_id = image_ids[trial][0][0].encode('UTF8')[:-4]  # Image-name in a weird unicode structure (Japan)
            for event in range(indexes_trial_on[trial], indexes_trial_off[trial]):
                event_label = msgs.label.values[event]
                if event_label in self.tag_dictionary:  # If it's in our dictionary we want to write it on file
                    event_time = msgs.index.values[event]
                    event_tag = event_dictionary[self.tag_dictionary[event_label]]
                    nextline = map(str, [event_time, task_id, self.recording, self.block, trial + 1, event_tag,
                                         image_id, trial_success])
                    image_lines.append(_delimiter.join(nextline))
        with open(_fileout, "w") as f:
            f.write('\n'.join(image_lines))

    def eyeevents2dat(self, _fileout, _delimiter='\t'):
        blinkdata = self.getblinks()
        saccdata = self.getsaccades(blinkdata)
        fixdata = self.getfixations()

        # Change (0,0) to center of the image
        saccdata['x_on'], saccdata['y_on'] = self.centercoordinates(saccdata['x_on'], saccdata['y_on'])
        saccdata['x_off'], saccdata['y_off'] = self.centercoordinates(saccdata['x_off'], saccdata['y_off'])
        fixdata['x_on'], fixdata['y_on'] = self.centercoordinates(fixdata['x_on'], fixdata['y_on'])
        fixdata['x_off'], fixdata['y_off'] = self.centercoordinates(fixdata['x_off'], fixdata['y_off'])
        fixdata['param1'], fixdata['param2'] = self.centercoordinates(fixdata['param1'], fixdata['param2'])

        # Convert data to degrees
        saccdata['x_on'], saccdata['y_on'] = self.pixels2degrees(saccdata['x_on'], saccdata['y_on'])
        saccdata['x_off'], saccdata['y_off'] = self.pixels2degrees(saccdata['x_off'], saccdata['y_off'])
        fixdata['x_on'], fixdata['y_on'] = self.pixels2degrees(fixdata['x_on'], fixdata['y_on'])
        fixdata['x_off'], fixdata['y_off'] = self.pixels2degrees(fixdata['x_off'], fixdata['y_off'])
        fixdata['param1'], fixdata['param2'] = self.pixels2degrees(fixdata['param1'], fixdata['param2'])

        eyeeventdata = np.append(saccdata, fixdata)
        eyeeventid = np.array([cms.eventsid['sacc']] * len(saccdata) + [cms.eventsid['fix']] * len(fixdata))
        idx_sort = eyeeventdata['on'].argsort()
        eyeeventdata = eyeeventdata[idx_sort]
        eyeeventid = eyeeventid[idx_sort]

        # generate output file
        eyevex_lines = []
        fields = ["eventID", ]
        fields.extend(eyeeventdata.dtype.names)
        eyevex_lines.append(_delimiter.join(fields))

        for evID, evdata in zip(eyeeventid, eyeeventdata):
            output = [evID, ]
            output.extend(evdata)
            eyevex_lines.append(_delimiter.join(map(str, output)))

        with open(_fileout, "w") as f:
            f.write('\n'.join(eyevex_lines))

    # A function to transform saccades in pixels to degree as Osaka data not in degrees
    # It is sugested to first centercoordinates() and then pixels2degrees()
    def pixels2degrees(self, _datax, _datay, _pxlperdeg=cms.pxlperdeg):
        datax = _datax / cms.pxlperdeg
        datay = _datay / cms.pxlperdeg
        return datax, datay

    # Make a filein and fileout list of every chosen task, subject and session in the root directory
    def makedatabaselist(self, tag='', indir=cms.rawdir, outdir=cms.outdir):
        _filesin = []
        _filesout = []
        for tsk in cms.selected_tasks:
            osakapath = "{dir}/Osaka/Human/{tsk}".format(dir=indir, tsk=tsk)
            subjects = os.listdir(osakapath)
            # Now iterate over subjects
            for sbj in subjects:
                if sbj not in cms.selected_subjects_osaka:
                    continue
                sesspath = "{dir}/{sbj}".format(dir=osakapath, sbj=sbj)
                sessions = os.listdir(sesspath)
                # Now iterate over sessions
                for sess in sessions:
                    if '.asc' not in sess:
                        continue
                    sessname = sess[:-4]  # Without the '.asc' part
                    _filesin.append("{dir}/{sess}".format(dir=sesspath, sess=sess))
                    _filesout.append("{dir}/Osaka_{tsk}_{sess}{tag}.dat".format(dir=outdir, sbj=sbj, tsk=tsk,
                                                                                sess=sessname, tag=tag))
        return _filesin, _filesout


class ChileSubject(SubjectType):
    # Chile notation of tags in asc file and it's conversion to OFFICIAL event tags (check commons)
    tag_dictionary = {'Imagen ploma': 'trial_start',
                      'imagen_natural': 'main_image_start',
                      'Ruido_rosa': 'main_image_end',
                      'BLANK': 'trial_end'
                      }

    def load(self, _filein=None):
        return ChileSubject(_filein)

    # Chile data structure: recording and block is always 1, get task from filepath
    def name2info(self):
        recording = 1
        block = 1
        taskname = re.search('Chile/([A-Za-z]+?)/', self.filepath).group(1)
        return recording, block, taskname

    def imageevents2dat(self, _fileout, _delimiter=','):
        header = cms.imageheader
        msgs = self.events.dframes['MSG']
        task_id = cms.tasksid[self.taskname]
        event_dictionary = cms.imagesid[cms.tasksid[self.taskname]]

        # Center X,Y samples
        self.samps.x_l, self.samps.y_l = self.centercoordinates(self.samps.x_l, self.samps.y_l)

        # Get success response from a message called 'precision' and image names from 'im_natural'
        response = []
        image_names = []
        for message in msgs.values:
            if len(message) == 3:
                submessage = message[2]
                if not isinstance(submessage, basestring):  # TODO: This warning is a pyCharm bug, wait for update
                    continue
                elif 'precision' in submessage:
                    response_number = int(submessage[-1:])
                    response.append(response_number)
                elif 'im_natural' in submessage:
                    image_name = submessage[-12:-4]
                    image_names.append(image_name)

        # We will write image events as lines info into a .dat file
        image_lines = [_delimiter.join(header)]
        image_lines.insert(0, 'task')  # This is just a format thing: first line must say "task"

        # Trial start and end indexes: start = trial ID, end = TRIAL_END, So we're containing a whole trial
        indexes_trial_on = np.where(msgs == 'Imagen ploma')[0]
        indexes_trial_off = np.where(msgs == 'BLANK')[0]
        for trial in range(len(indexes_trial_off)):
            for event in range(indexes_trial_on[trial], indexes_trial_off[trial]):
                event_labels = msgs.values[event]
                # If label is in dictionary we append a line
                if len(event_labels) >= 3 and (event_labels[2] in self.tag_dictionary):
                    event_time = msgs.index.values[event]
                    event_tag = event_dictionary[self.tag_dictionary[event_labels[2]]]
                    next_line = map(str, [event_time, task_id, self.recording, self.block, trial + 1, event_tag,
                                          image_names[trial], response[trial]])
                    image_lines.append(_delimiter.join(next_line))
        with open(_fileout, "w") as f:
            f.write('\n'.join(image_lines))

    def eyeevents2dat(self, _fileout, _delimiter='\t'):
        blinkdata = self.getblinks()
        saccdata = self.getsaccades(blinkdata)
        fixdata = self.getfixations()

        # Change (0,0) to center of the image
        saccdata['x_on'], saccdata['y_on'] = self.centercoordinates(saccdata['x_on'], saccdata['y_on'])
        saccdata['x_off'], saccdata['y_off'] = self.centercoordinates(saccdata['x_off'], saccdata['y_off'])
        fixdata['x_on'], fixdata['y_on'] = self.centercoordinates(fixdata['x_on'], fixdata['y_on'])
        fixdata['x_off'], fixdata['y_off'] = self.centercoordinates(fixdata['x_off'], fixdata['y_off'])
        fixdata['param1'], fixdata['param2'] = self.centercoordinates(fixdata['param1'], fixdata['param2'])

        eyeeventdata = np.append(saccdata, fixdata)
        eyeeventid = np.array([cms.eventsid['sacc']] * len(saccdata) + [cms.eventsid['fix']] * len(fixdata))
        idx_sort = eyeeventdata['on'].argsort()
        eyeeventdata = eyeeventdata[idx_sort]
        eyeeventid = eyeeventid[idx_sort]

        # generate output file
        eyevex_lines = []
        fields = ["eventID", ]
        fields.extend(eyeeventdata.dtype.names)
        eyevex_lines.append(_delimiter.join(fields))

        for evID, evdata in zip(eyeeventid, eyeeventdata):
            output = [evID, ]
            output.extend(evdata)
            eyevex_lines.append(_delimiter.join(map(str, output)))

        with open(_fileout, "w") as f:
            f.write('\n'.join(eyevex_lines))

    def makedatabaselist(self, tag='', indir=cms.rawdir, outdir=cms.outdir):
        _filesin = []
        _filesout = []
        for _tsk in cms.selected_tasks:
            _chilepath = "{dir}/Chile/{tsk}".format(dir=indir, tsk=_tsk)
            _subjects = os.listdir(_chilepath)
            for _sbj in _subjects:
                if _sbj not in cms.selected_subjects_chile:
                    continue
                _filesin.append("{dir}/{sbj}/{sbj}.asc".format(dir=_chilepath, sbj=_sbj))
                _filesout.append("{dir}/Chile_{tsk}_{sbj}{tag}.dat".format(dir=outdir, tsk=_tsk, sbj=_sbj, tag=tag))
        return _filesin, _filesout


# MAIN ROUTINE: Make Osaka and Chile processed database (from asc to python compatible *.dat files)
if __name__ == '__main__':
    # Chile subjects
    print "Processing Chile Subjects"
    aSubject = ChileSubject()
    filesin, filesout = aSubject.makedatabaselist('_task')
    aSubject.imageevents2dats(filesin, filesout)
    filesin, filesout = aSubject.makedatabaselist('_eye')
    aSubject.eyeevents2dats(filesin, filesout)
    # Osaka subjects
    print "Processing Osaka Subjects"
    aSubject = OsakaSubject()
    filesin, filesout = aSubject.makedatabaselist('_task')
    aSubject.imageevents2dats(filesin, filesout)
    filesin, filesout = aSubject.makedatabaselist('_eye')
    aSubject.eyeevents2dats(filesin, filesout)
