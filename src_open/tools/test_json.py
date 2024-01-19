import json
import logging

logger = logging.getLogger(__name__)


def read_json(file_path):
    with open(file_path, 'r') as f:
        file_dict = json.load(f)
    return file_dict

def test_json(cfg):
    assert ('load_json' in cfg)
    assert ('obj_names' in cfg)
    assert ('slices' in cfg)

    metrics_total_5cm_5d = 0
    metrics_total_2cm_2d = 0
    metrics_total_ADD_01d = 0
    metrics_total_ADD_005d = 0
    metrics_total_ADD_002d = 0
    total_num = 0

    metrics = read_json(cfg.load_json)
    for slice in cfg.slices:
        metrics_5cm_5d = 0
        metrics_2cm_2d = 0
        metrics_ADD_01d = 0
        metrics_ADD_005d = 0
        metrics_ADD_002d = 0
        num = 0
        
        for obj in cfg.obj_names:
            obj_slice = str(obj) + '_' + str(slice)
            metrics_obj_slice = metrics[obj_slice]
            metrics_5cm_5d_obj_slice = metrics_obj_slice['5cm_5d']
            metrics_2cm_2d_obj_slice = metrics_obj_slice['2cm_2d']
            metrics_ADD_01d_obj_slice = metrics_obj_slice['ADD_0.1d']
            metrics_ADD_005d_obj_slice = metrics_obj_slice['ADD_0.05d']
            metrics_ADD_002d_obj_slice = metrics_obj_slice['ADD_0.02d']
            num_obj_slice = metrics_obj_slice['num']

            metrics_5cm_5d += metrics_5cm_5d_obj_slice * num_obj_slice
            metrics_2cm_2d += metrics_2cm_2d_obj_slice * num_obj_slice
            metrics_ADD_01d += metrics_ADD_01d_obj_slice * num_obj_slice
            metrics_ADD_005d += metrics_ADD_005d_obj_slice * num_obj_slice
            metrics_ADD_002d += metrics_ADD_002d_obj_slice * num_obj_slice
            num += num_obj_slice

        metrics_total_5cm_5d += metrics_5cm_5d
        metrics_total_2cm_2d += metrics_2cm_2d
        metrics_total_ADD_01d += metrics_ADD_01d
        metrics_total_ADD_005d += metrics_ADD_005d
        metrics_total_ADD_002d += metrics_ADD_002d
        total_num += num

        metrics_avg_5cm_5d = metrics_5cm_5d / num
        metrics_avg_2cm_2d = metrics_2cm_2d / num
        metrics_avg_ADD_01d = metrics_ADD_01d / num
        metrics_avg_ADD_005d = metrics_ADD_005d / num
        metrics_avg_ADD_002d = metrics_ADD_002d / num

        logger.info(f"Metrics report for slice {slice}")
        logger.info(f"Metrics average 5cm 5d {metrics_avg_5cm_5d}")
        logger.info(f"Metrics average 2cm 2d {metrics_avg_2cm_2d}")
        logger.info(f"Metrics average ADD 0.1d {metrics_avg_ADD_01d}")
        logger.info(f"Metrics average ADD 0.05d {metrics_avg_ADD_005d}")
        logger.info(f"Metrics average for 0.02d {metrics_avg_ADD_002d}")

    metrics_total_avg_5cm_5d = metrics_total_5cm_5d / total_num
    metrics_total_avg_2cm_2d = metrics_total_2cm_2d / total_num
    metrics_total_avg_ADD_01d = metrics_total_ADD_01d / total_num
    metrics_total_avg_ADD_005d = metrics_total_ADD_005d / total_num
    metrics_total_avg_ADD_002d = metrics_total_ADD_002d / total_num

    logger.info(f"Metrics report for all slices")
    logger.info(f"num {total_num}")
    logger.info(f"Metrics average 5cm 5d {metrics_total_avg_5cm_5d}")
    logger.info(f"Metrics average 2cm 2d {metrics_total_avg_2cm_2d}")
    logger.info(f"Metrics average ADD 0.1d {metrics_total_avg_ADD_01d}")
    logger.info(f"Metrics average ADD 0.05d {metrics_total_avg_ADD_005d}")
    logger.info(f"Metrics average for 0.02d {metrics_total_avg_ADD_002d}")    


def main(cfg):
    globals()['test_'+cfg.task](cfg)