import numpy as np

class PingData:
    def __init__(self, num_tx, num_rx):
        # Going to assume that tx fire cyclically and rx listen for each tx
        self.num_tx = num_tx
        self.num_rx = num_rx

        self.pings = []

    def create_ping_list(self, raw_data, mf_data):
        assert raw_data.dtype == np.complex128

        (num_samples, num_rx_data) = raw_data.shape
        # Each ping is num_tx samples
        for count, i in enumerate(range(0, num_rx_data, self.num_rx)):
            ping = {}
            raw_channels = raw_data[:, i:i+self.num_rx]
            mf_channels = mf_data[:, i:i+self.num_rx]

            ping['tx_id'] = (count % self.num_tx) + 1
            ping['rx_raw'] = raw_channels
            ping['mf_raw'] = mf_channels

            self.pings.append(ping)



