import constants as c

"""
This is the master class that contains everything needed for scene reconstruction.
"""
class SASDataSchema:
    def __init__(self):
        self._data = {
            c.WFM_DATA: None, # Raw measurements, size num_sensor_pos x num_samples, s(t)
            c.RX_COORDS: None, # position of receivers in meters, size num_sensor_pos x 3
            c.TX_COORDS: None, # position of transmitters in meters, size num_sensor_pos x 3 (same size as rx_coords)
            c.RX_VECS: None, # direction vector for each receiver, size num_sensor_pos x 3
            c.TX_VECS: None, # direction vector for each transmitter, size num_sensor_pos x 3
            c.TEMPS: None, # temp of each measurement, size num_sensor_pos x 1
            c.SYS_PARAMS: None, # points to a class SysParams instantiation (see classes below)
            c.WFM_PARAMS: None, # points to a class WfmParams instantiation (see classes below)
            c.WFM: None, # the transmitted pulse, p(t)
            c.GEOMETRY: None, # points to a class Geometry instantiation (see classes below)
            c.WFM_RC: None, # the matched filter waveforms, size num_sensor_pos x num_samples, s_MF(t)
            c.WFM_CROP_SETTINGS: None, # points to a class WfmCropSettings instantiation (see classes below)
            c.SOUND_SPEED: None, # sound speed (usually computed using the mean temperature and physics model)
            c.SAME_TX_PER_K_RX: None # used for speedup if the same tx is used for multiple rx (as is the case in svss)
        }

    def __str__(self):
        return str(self._data)

    def __getitem__(self, key):
        try:
            value = self._data[key]
            assert value is not None, ' '.join(["Value for", str(key), "has not been set"])
            return value
        except KeyError:
            raise KeyError("Key is not valid")

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except KeyError:
            raise KeyError("Key is not valid")

    def validate_schema(self):
        # Check wfm data
        assert self._data[c.WFM_DATA].ndim == 2, "WFM data should be 2 dimensional array [num_pings, num_samples]"
        # Check rx coords
        assert self._data[c.RX_COORDS].ndim == 2, "RX coords should be 2 dimensional array [num_pings, 3]"
        # check tx coords
        assert self._data[c.TX_COORDS].ndim == 2, "RX coords should be 2 dimensional array [num_pings, 3]"

        # Validate rx/tx coord shape
        assert self._data[c.RX_COORDS].shape[0] == self._data[c.TX_COORDS].shape[0], "RX coords num pings should match " \
                                                                   "tx coords num pings"
        # Make sure we have enough waveforms
        assert self._data[c.WFM_DATA].shape[0] == self._data[c.RX_COORDS].shape[0], "Number of waveforms should " \
                                                                  "match number of rx/tx positions"

        raise NotImplementedError("Make sure user has everything they need to reconstruct")

"""
This is a helper class that is referenced SASDataSchema. It contains info related to the transducer array
"""
class SysParams:
    def __init__(self):
        self._data = {
            c.TX_POS: None, # this is the starting position of the first tx array, only used for airsas
            c.RX_POS: None, # this is the starting position of the first rx element, only used for airsas
            c.TX_BW: None, # this is the beamwidth of the tx elements
            c.RX_BW: None, # this is the beamwidth of the rx elements
            c.CENTER: None, # this is the scene center relative to the starting tx and rx positions
            c.GROUP_DELAY: None, # this is the group delay of the digitizing hardware
            c.FS: None, # sample rate
            c.FC: None # center frequency
        }

    def __str__(self):
        return str(self._data)

    def __getitem__(self, key):
        try:
            value = self._data[key]
            assert value is not None, ' '.join(["Value for", str(key), "has not been set"])
            return value
        except KeyError:
            raise KeyError("Key is not valid")

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except KeyError:
            raise KeyError("Key is not valid")

    def validate_schema(self):
        raise NotImplementedError("Make sure user has everything they need to reconstruct")

"""
This is a helper class that is referenced SASDataSchema. It contains info related to transmitted LFM pulse p(t)
"""
class WfmParams:
    def __init__(self):
        self._data = {
            c.F_START: None, # start frequency of LFM (HZ)
            c.F_STOP: None, # stop frequency of LFM (HZ)
            c.T_DUR: None, # duration of LFM (s)
            c.WIN_RATIO: None # ratio of Tukey window
        }

    def __str__(self):
        return str(self._data)

    def __getitem__(self, key):
        try:
            value = self._data[key]
            assert value is not None, ' '.join(["Value for", str(key), "has not been set"])
            return value
        except KeyError:
            raise KeyError("Key is not valid")

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except KeyError:
            raise KeyError("Key is not valid")

    def validate_schema(self):
        raise NotImplementedError("Make sure user has everything they need to reconstruct")

"""
This is a helper class that is referenced SASDataSchema. We want to sample time coordinates that correspond
to ellipsoids within the scene bounds. This class helps do that by cropping measurements to the scene dimensions. 
In particular, given measurements that are x[0:number_of_samples] long, we crop them
to x[MIN_SAMPLE:MIN_SAMPLE + NUM_SAMPLES], where MIN_SAMPLE and NUM_SAMPLE are computed using knowledge of the 
transducer elements with respect to the scene. 
"""
class WfmCropSettings:
    def __init__(self):
        self._data = {
            c.MIN_SAMPLE: None,
            c.MIN_DIST: None,
            c.MAX_DIST: None,
            c.NUM_SAMPLES: None
        }

    def __str__(self):
        return str(self._data)

    def __getitem__(self, key):
        try:
            value = self._data[key]
            assert value is not None, ' '.join(["Value for", str(key), "has not been set"])
            return value
        except KeyError:
            raise KeyError("Key is not valid")

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except KeyError:
            raise KeyError("Key is not valid")

    def validate_schema(self):
        raise NotImplementedError("Make sure user has everything they need to reconstruct")


"""
This is a helper class that is referenced SASDataSchema. It is used to define the scene boundaries and voxels used
for backprojection. 
"""
class Geometry:
    def __init__(self):
        self._data = {
            c.CORNERS: None, # The corners of the scene in meters, size [8 x 3]
            c.VOXELS: None, # the scene positions in meters, size [num_voxels, 3]
            c.NUM_X: None, # the number of x voxels
            c.NUM_Y: None, # the number of y voxels
            c.NUM_Z: None, # the number of z voxels
            c.X_DIM: None, # abs(x_max - x_min)
            c.Y_DIM: None, # abs(y_max - y_min)
            c.Z_DIM: None # abs(z_max - z_min)
        }

    def __str__(self):
        return str(self._data)

    def __getitem__(self, key):
        try:
            value = self._data[key]
            assert value is not None, ' '.join(["Value for", str(key), "has not been set"])
            return value
        except KeyError:
            raise KeyError("Key is not valid")

    def __setitem__(self, key, value):
        try:
            self._data[key] = value
        except KeyError:
            raise KeyError("Key is not valid")

    def validate_schema(self):
        raise NotImplementedError("Make sure user has everything they need to reconstruct")