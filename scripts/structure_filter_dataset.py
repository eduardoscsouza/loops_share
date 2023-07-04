import os
import sys
import shutil

import pandas as pd

sys.path.append(os.path.join(os.environ["LOOPS_PATH"], "src"))
from utils import *



if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(os.environ["LOOPS_PATH"], "data", "raw")
    filtered_csv = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(os.environ["LOOPS_PATH"], "data", "filtered_list.csv")
    filtered_output_dir = sys.argv[3] if len(sys.argv) >= 4 else os.path.join(os.environ["LOOPS_PATH"], "data", "filtered")
    unfiltered_output_dir = sys.argv[4] if len(sys.argv) >= 5 else os.path.join(os.environ["LOOPS_PATH"], "data", "unfiltered")
    removed_output_dir = sys.argv[5] if len(sys.argv) >= 6 else os.path.join(os.environ["LOOPS_PATH"], "data", "removed")
    input_test_dir = sys.argv[6] if len(sys.argv) >= 7 else os.path.join(os.environ["LOOPS_PATH"], "data", "raw", "Test")
    test_csv = sys.argv[7] if len(sys.argv) >= 8 else os.path.join(os.environ["LOOPS_PATH"], "data", "test_set.csv")

    shutil.copytree(os.path.join(input_dir, NORMAL_CLASS), os.path.join(unfiltered_output_dir, NORMAL_CLASS))
    shutil.copytree(os.path.join(input_dir, POTENTIALLY_MALIGNANT_CLASS), os.path.join(unfiltered_output_dir, POTENTIALLY_MALIGNANT_CLASS))
    shutil.copytree(os.path.join(input_dir, MALIGNANT_CLASS), os.path.join(unfiltered_output_dir, MALIGNANT_CLASS))

    test_dir = os.path.join(unfiltered_output_dir, "test_dir")
    shutil.copytree(input_test_dir, test_dir)

    test_imgs = []
    test_xmls = sorted(glob(os.path.join(test_dir, "**", "*.xml"), recursive=True))
    for xml in test_xmls:
        img = get_xml_matching_imgs(xml)
        if len(img) == 0:
            print_error(xml, "No Matching Images Found")
            continue
        if len(img) > 1:
            print_error(xml, "Multiple Matching Images Found")
        img = img[0]

        img = get_clean_sample_path(img, test_dir)
        xml = get_clean_sample_path(xml, test_dir)
        os.rename(os.path.join(test_dir, img), os.path.join(unfiltered_output_dir, img))
        os.rename(os.path.join(test_dir, xml), os.path.join(unfiltered_output_dir, xml))

        test_imgs += [img]

    pd.DataFrame(test_imgs, columns=["Image"]).to_csv(test_csv, index=False)
    for test_subdir, _, _ in reversed(list(os.walk(test_dir))):
        os.rmdir(test_subdir)

    df = pd.read_csv(filtered_csv)
    shutil.copytree(unfiltered_output_dir, filtered_output_dir)
    for img_path in df["Removed File"]:
        removed_file = os.path.join(filtered_output_dir, img_path)
        removed_file_dest = os.path.join(removed_output_dir, img_path)
        xml = "{}.xml".format(os.path.splitext(removed_file)[0])
        xml_dest = "{}.xml".format(os.path.splitext(removed_file_dest)[0])

        os.makedirs(os.path.dirname(removed_file_dest), exist_ok=True)
        os.rename(removed_file, removed_file_dest)
        if os.path.exists(xml):
            os.rename(xml, xml_dest)
