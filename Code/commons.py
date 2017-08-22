# This file contains commonly used variables and directories

# My root directory where I save both raw and processed data
database = "D:/Drive/Database/SearchMemory"

# Raw directories of Osaka and Chile database
rawdir = "{dir}/RAW".format(dir=database)

# Directory where to save output databse
outdir = "{dir}/Processed".format(dir=database)

# Can select many file groups to process at the same time: 'Search', 'Memory' are currently what we have
selected_tasks = {'Search', 'Memory', }

# Osaka subjects in database, rejected ones are greyed out
selected_subjects_osaka = {

    "FM",
    "GH",
    "KH",
    "KO",
    "MF",
    "MH",
    "MN",
    "NM",
    "RI",
    "RO",
    "TU",
    "YM",
}

# Chile subjects in database, rejected ones are greyed out (Greyed out may mean data is corrupted in some way)
selected_subjects_chile = {

    "camic",
    "camica",
    "caroc",
    # "cata",
    "ceci",
    "claudio",
    # "dani",
    "diego",
    "diegob",
    "diegog",
    "esteban",
    "ignacio",
    # "jd",
    "luism",
    "marie",
    "matir",
    # "max",
    # "michi",
    "miguel",
    "negro",
    # "nestor",
    "nicoc",
    # "nicom",
    # "noelia",
    "pablo",
    "patienc",
    # "rebe",
    # "roci",
    "rocio",
    # "rodri",
    "samuel",
    # "shintaro",

}

# Dictionary containing events ID
eventsid = {'sacc': 100,
            'fix': 200,
            'blink': 300,
            }

# Dictionary containing tasks ID
tasksid = {'Global': 0,
           'Fixation': 1,
           'Calibration': 2,
           'Free Viewing': 3,
           'Search': 4,
           'Memory': 5
           }
# Array of dictionaries containing image-events ID for each task
# TODO: Check Yukako's pdf presentation to see explanation and check if this can be optimized
imagesid = [dict() for x in range(len(tasksid))]
imagesid[tasksid['Global']] = {'reset_trial': 0
                               }

imagesid[tasksid['Fixation']] = {'trial_start': 101,  # Same as fixpoint_on
                                 'eye_in': 102,
                                 'no_eye': -101,
                                 'fail_fix_window': -102,
                                 'fix_complete': 103,
                                 'juice_delivery': 104,
                                 'trial_end': 105,
                                 'fix_point_off': 110
                                 }

imagesid[tasksid['Calibration']] = {'trial_start': 201,
                                    'eye_in': 202,
                                    'no_eye': -201,
                                    'fail_fix_window': -202,
                                    'fix_complete': 203,
                                    'juice_delivery': 204,
                                    'trial_end': 205,
                                    'fix_point_off': 210
                                    }

imagesid[tasksid['Free Viewing']] = {'trial_start': 301,
                                     'eye_in': 302,
                                     'no_eye': -301,
                                     'fail_fix_window': -302,
                                     'fix_complete': 303,
                                     'fail_out_image': -303,
                                     'complete_image_viewing': 304,
                                     'juice_delivery': 305,
                                     'trial_end': 306,
                                     'fix_point_off': 310,
                                     'main_image_start': 311,
                                     'main_image_end': 312
                                     }

imagesid[tasksid['Search']] = {'trial_start': 401,
                               'eye_in': 402,
                               'no_eye': -401,
                               'fail_fix_window': -402,
                               'fix_complete': 403,
                               'fail_out_image': -403,
                               'complete_image_viewing': 404,
                               'juice_delivery': 405,
                               'lever_fail_window': -404,
                               'out_image': -405,
                               'time_up_no_lever': -406,
                               'failed_keep_lever': -407,
                               'failed_eye_on_target': -408,
                               'complete_answer': 406,
                               'trial_end': 408,
                               'fix_point_off': 410,
                               'main_image_start': 411,
                               'main_image_end': 412,
                               'object_image_start': 413,
                               'object_image_end': 414,
                               }

imagesid[tasksid['Memory']] = {'trial_start': 501,
                               'eye_in': 502,
                               'no_eye': -501,
                               'fail_fix_window': -502,
                               'fix_complete': 503,
                               'fail_out_image': -503,
                               'complete_image_viewing': 504,
                               'correct_answer_lever': 505,
                               'false_alarm_lever_press': -504,
                               'correct_answer_no_lever': 506,
                               'false_answer_no_lever': -505,
                               'juice_delivery': 507,
                               'trial_end': 508,
                               'fix_point_off': 510,
                               'main_image_start': 511,
                               'main_image_end': 512,
                               'object_image_start': 513,
                               'object_image_end': 514
                               }

imageheader = ("TIMING_CLOCK", "g_task_switch", "g_rec_no", "g_block_num", "TRIAL_NUM", "log_task_ctrl", "t_tgt_data",
               "SF_FLG")

# Osaka's pixel per degree conversion
pxlperdeg = 29.69
