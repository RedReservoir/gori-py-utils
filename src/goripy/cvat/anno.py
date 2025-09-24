import numpy

import xml



def compute_anno_idx_to_job_num_arr(root_anno_xml_item):
    """
    Computes the Job ID of each Annotation ID from a CVAT format annotation file.

    Args:

        root_anno_xml_item (xml.etree.ElementTree.ElementTree):
            Root XML node of the CVAT annotation file.

    Returns:
    
        numpy.ndarray:
            A 1D numpy array where, for each Annotation ID (index), the Job ID is stored (value).
    """

    meta_xml_item = root_anno_xml_item.find("meta")
    task_xml_item = meta_xml_item.find("task")

    num_anno_items = int(task_xml_item.find("stop_frame").text) + 1
    anno_idx_to_job_num_arr = numpy.empty(shape=(num_anno_items), dtype=int)

    for seg_xml_item in task_xml_item.find("segments").findall("segment"):
        
        job_num = int(seg_xml_item.find("id").text)
        start_id = int(seg_xml_item.find("start").text)
        end_id = int(seg_xml_item.find("stop").text)

        anno_idx_to_job_num_arr[start_id:end_id+1] = job_num
    
    return anno_idx_to_job_num_arr
