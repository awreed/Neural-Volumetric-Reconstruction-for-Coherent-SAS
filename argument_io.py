import constants as c
import glob
import os
import shutil

def directory_cleaup(output_dir, clear_output_directory):
    # Delete contents of output_directory so we can download fresh stuff only
    if clear_output_directory:
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            shutil.rmtree(f, ignore_errors=True)

            #try:
            #    os.remove(f)
            #except IsADirectoryError:
            #
            #except OSError:
            #    shutil.rmtree(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(os.path.join(output_dir, c.IMAGES)):
        os.makedirs(os.path.join(output_dir, c.IMAGES), exist_ok=True)

    if not os.path.exists(os.path.join(output_dir, c.NUMPY)):
        os.makedirs(os.path.join(output_dir, c.NUMPY), exist_ok=True)