import os
import logging
import argparse
from pathlib import Path

import numpy as np
import dbdicom as db
import miblab_dl as dl
import torch

from ibeat_kidney_shape.utils import data, edit

MODULE_DIR = Path(__file__).resolve().parent

# These need fully manual segmentation
EXCLUDE = [ 
    '4128_055', # miblab nnunet No segmentation: large left kidney and tiny right kidney
    '7128_149', # miblab nnunet Segmentation failed: horseshoe kidney
]

# Exceptions: failed with nnunet for no obvious reason
TOTSEG = [
    ('7128_085', 'Baseline'),

    # Leeds
    ('4128_007', 'Baseline'),
    ('4128_010', 'Baseline'), # poor images
    ('4128_012', 'Baseline'),
    ('4128_013', 'Baseline'), # poor images
    ('4128_014', 'Baseline'),
    ('4128_015', 'Baseline'),
    ('4128_016', 'Baseline'),
    ('4128_017', 'Baseline'),
    ('4128_024', 'Baseline'),
    ('4128_030', 'Baseline'),
    ('4128_043', 'Baseline'),
    ('4128_051', 'Baseline'),
    ('4128_052', 'Baseline'),
    ('4128_053', 'Baseline'),
    ('4128_054', 'Baseline'),
    ('4128_061', 'Baseline'),

    # Sheffield
    ('7128_021', 'Baseline'),
    ('7128_026', 'Baseline'),
    ('7128_027', 'Baseline'),
    ('7128_033', 'Baseline'),
    ('7128_037', 'Baseline'),
    ('7128_038', 'Baseline'),
    ('7128_040', 'Baseline'),
    ('7128_044', 'Baseline'),
    ('7128_047', 'Baseline'),
    ('7128_055', 'Baseline'), # failed with nnunet
    ('7128_056', 'Baseline'),
    ('7128_059', 'Baseline'),
    ('7128_064', 'Baseline'),
    ('7128_067', 'Baseline'),
    ('7128_069', 'Baseline'),
    ('7128_072', 'Baseline'),
    ('7128_073', 'Baseline'),
    ('7128_074', 'Baseline'),
    ('7128_075', 'Baseline'),
    ('7128_076', 'Baseline'),
    ('7128_077', 'Baseline'),
    ('7128_080', 'Baseline'),
    ('7128_081', 'Baseline'),
    ('7128_082', 'Baseline'),
    ('7128_083', 'Baseline'),
    ('7128_084', 'Baseline'),
    ('7128_086', 'Baseline'),
    ('7128_087', 'Baseline'),
    ('7128_091', 'Baseline'),
    ('7128_092', 'Baseline'),
    ('7128_093', 'Baseline'),
    ('7128_094', 'Baseline'),
    ('7128_096', 'Baseline'),
    ('7128_101', 'Baseline'),
    ('7128_102', 'Baseline'),
    ('7128_104', 'Baseline'),
    ('7128_106', 'Baseline'),
    ('7128_109', 'Baseline'),
    ('7128_110', 'Baseline'),
    ('7128_111', 'Baseline'),
    ('7128_112', 'Baseline'),
    ('7128_113', 'Baseline'),
    ('7128_114', 'Baseline'), # very poor images
    ('7128_115', 'Baseline'),
    ('7128_116', 'Baseline'),
    ('7128_117', 'Baseline'),
    ('7128_118', 'Baseline'),
    ('7128_129', 'Baseline'),
    ('7128_132', 'Baseline'),
    ('7128_137', 'Baseline'),
    ('7128_140', 'Baseline'),
    ('7128_144', 'Baseline'),
    ('7128_146', 'Baseline'),
    ('7128_147', 'Baseline'),
    ('7128_148', 'Baseline'),
    ('7128_155', 'Baseline'),
    ('7128_156', 'Baseline'),
    ('7128_157', 'Baseline'),
    ('7128_160', 'Baseline'),
    ('7128_163', 'Baseline'),
    ('7128_164', 'Baseline'),
    ('7128_165', 'Baseline'),
    ('7128_166', 'Baseline'),

    ('2128_007', 'Baseline'),
    ('2128_009', 'Baseline'),
    ('2128_020', 'Baseline'),
    ('2128_028', 'Baseline'),
    ('2128_032', 'Baseline'),
    ('2128_040', 'Baseline'),
    ('2128_045', 'Baseline'),
    ('6128_001', 'Baseline'),
    ('6128_001', 'Followup'),
    ('6128_009', 'Baseline'),

    ('3128_007', 'Followup'),
    ('3128_014', 'Baseline'),
    ('3128_014', 'Followup'),
    ('3128_018', 'Baseline'),
    ('3128_019', 'Baseline'),
    ('3128_019', 'Followup'),
    ('3128_023', 'Baseline'),
    ('3128_024', 'Baseline'),
    ('3128_026', 'Baseline'),
    ('3128_026', 'Followup'),
    ('3128_031', 'Baseline'),
    ('3128_033', 'Followup'),
    ('3128_043', 'Baseline'),
    ('3128_044', 'Baseline'),
    ('3128_045', 'Baseline'),
    ('3128_047', 'Baseline'),
    ('3128_050', 'Baseline'),
    ('3128_056', 'Baseline'),
    ('3128_058', 'Baseline'),
    ('3128_059', 'Baseline'),
    ('3128_067', 'Baseline'),
    ('3128_067', 'Followup'),
    ('3128_070', 'Baseline'),
    ('3128_074', 'Baseline'),
    ('3128_074', 'Followup'),
    ('3128_078', 'Followup'),
    ('3128_080', 'Baseline'),
    ('3128_081', 'Baseline'),
    ('3128_086', 'Baseline'),
    ('3128_086', 'Followup'),
    ('3128_090', 'Baseline'),
    ('3128_091', 'Baseline'),
    ('3128_091', 'Followup'),
    ('3128_095', 'Baseline'),
    ('3128_095', 'Followup'),
    ('3128_096', 'Baseline'),
    ('3128_096', 'Followup'),
    ('3128_097', 'Baseline'),
    ('3128_098', 'Baseline'),
    ('3128_107', 'Baseline'),
    ('3128_114', 'Baseline'),
    ('3128_115', 'Baseline'),
    ('3128_116', 'Baseline'),
    ('3128_131', 'Baseline'),
    ('3128_133', 'Baseline'),
    ('3128_137', 'Baseline'),
]


def run(build):

    logging.basicConfig(
        filename=os.path.join(build, 'kidney_shape', 'stage_1_segment.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    dir_data = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data') 
    dir_output = os.path.join(build, 'kidney_shape', 'stage_1_segment') 
    os.makedirs(dir_output, exist_ok=True)

    for database in ['Controls', 'Patients']:
        db_data = os.path.join(dir_data, database) 
        db_masks = os.path.join(dir_output, database) 
        run_db(db_data, db_masks)


def run_db(db_data, db_masks):

    os.makedirs(db_masks, exist_ok=True)

    # List of selected dixon series
    record = data.dixon_record(MODULE_DIR)

    # Get out phase series
    series = db.series(db_data)
    series_out_phase = [s for s in series if s[3][0][-9:]=='out_phase']

    # Loop over the out-phase series
    for series_op in series_out_phase:

        # Descriptors
        patient = series_op[1]
        study = series_op[2][0]
        sequence = series_op[3][0][:-10]
        selected_sequence = data.dixon_series_desc(record, patient, study)

        # Input and output series
        series_ip = series_op[:3] + [(sequence + '_in_phase', 0)]
        series_wi = series_op[:3] + [(sequence + '_water', 0)]
        series_fi = series_op[:3] + [(sequence + '_fat', 0)]
        mask_series = [db_masks, patient, (study, 0), (f'kidney_masks', 0)]

        # Skip if it is not the right sequence
        if sequence != selected_sequence:
            continue

        # Skip if the kidney masks already exist
        if db.exists(mask_series):
            continue
        
        try:
            # Read the in- and out of phase volumes
            op = db.volume(series_op)
            ip = db.volume(series_ip)

            # If autosegmentation does not work, draw rectangles in the middle slice 
            if patient in EXCLUDE:
                label_array = np.zeros(op.shape, dtype=np.int16)
                xm = int(np.round(op.shape[0]/2))
                ym = int(np.round(op.shape[1]/2))
                zm = int(np.round(op.shape[2]/2))
                label_array[xm+1:xm+10, ym+1:ym+10, zm] = 1
                label_array[xm-10:xm-1, ym-10:ym-1, zm] = 2

            elif (patient, study) in TOTSEG: # Total segmentator
                device = 'gpu' if torch.cuda.is_available() else 'cpu'
                label_vol = dl.totseg(op, cutoff=0.01, task='total_mr', device=device)
                label_array = label_vol.values
                # Extract kidneys only
                label_array[~np.isin(label_array, [2,3])] = 0
                # Relabel left and right
                label_array[label_array==3] = 1
                # Remove smaller disconnected clusters
                label_array = edit.largest_cluster_label(label_array)

            else: # nnunet
                # Read fat and water data
                wi = db.volume(series_wi)
                fi = db.volume(series_fi)
                # Compute label
                array = np.stack((op.values, ip.values, wi.values, fi.values), axis=-1)
                label_array = dl.kidney_pc_dixon(array, verbose=True)
            
            # Save output and log success
            db.write_volume((label_array, op.affine), mask_series, ref=series_op)
            logging.info(f"Successfully segmented {patient}, {study}.")

        except:

            logging.exception(f"Error segmenting {patient}, {study}.")

     

if __name__=='__main__':

    BUILD = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)
            