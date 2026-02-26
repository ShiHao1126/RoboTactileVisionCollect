import os
import csv
import time
import numpy as np
import spdlog

logger = spdlog.ConsoleLogger('FTSensor')

class FTSensor:    
    def __init__(
        self, 
        csv_save_path: str = None,
        data_header: list = [
            'image_name',  # image file name
            'timestamp',
            'fx', 'fy', 'fz',  # force (N)
            'mx', 'my', 'mz',  # torque (Nm)
            'x_mm', 'y_mm'   # sampling point (mm)
        ]
        ):
        assert csv_save_path is not None, "[FTSensor] csv_save_path is required"
        self.csv_save_path = csv_save_path
        self.data_header = data_header

        self._write_header()

    def _write_header(self):
        if not os.path.exists(self.csv_save_path):
            os.makedirs(os.path.dirname(self.csv_save_path), exist_ok=True)
            with open(self.csv_save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.data_header)
    
    def write_in(self, save_data: dict, timestamp: float = None):
        if timestamp is not None:
            save_data['timestamp'] = timestamp
        
        row_data = []
        missing_keys = []
        
        for key in self.data_header:
            if key in save_data:
                value = save_data[key]
                if value is None:
                    row_data.append('None')
                elif isinstance(value, (list, np.ndarray)):
                    # list or array -> string (comma separated)
                    row_data.append(','.join(str(v) for v in value))
                else:
                    row_data.append(str(value))
            else:
                row_data.append('KeyNotFound')
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(
                f"Missing keys in save_data: {missing_keys}. "
            )
        
        with open(self.csv_save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)


    def release(self):
        pass