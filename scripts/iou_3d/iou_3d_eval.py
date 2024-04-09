import os
import sys
import type_0
import type_1
import type_2
import type_3
import type_4

procsssor = {
    "type_0": (type_0.without_bias, type_0.with_bias),
    "type_1": (type_1.without_bias, type_1.with_bias),
    "type_2": (type_2.without_bias, type_2.with_bias),
    "type_3": (type_3.without_bias, type_3.with_bias),
    "type_4": (type_4.without_bias, type_4.with_bias),
}

def check_and_make(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    roots = [""]
    data_root = "./data"
    for root in roots:
        global_iou_3d = 0.0
        global_max_iou_3d = 0.0
        global_min_iou_3d = 1 << 30
        iou_3d_root = os.path.join(data_root, "iou_3d_results", root)
        prediction_line_root = os.path.join(data_root, "prediction_results", root)
        check_and_make(iou_3d_root)
        for j in range(0, 5):
            iou_3d = 0.0
            cnt = 0
            error = []
            iou_3d_prt_type_path = os.path.join(iou_3d_root, f"iou_3d_prt\\type_{j}")
            iou_3d_txt_type_path = os.path.join(iou_3d_root, f"iou_3d_txt\\type_{j}")
            prediction_line_type_path = os.path.join(prediction_line_root, f"prediction_line\\type_{j}")
            check_and_make(iou_3d_prt_type_path)
            check_and_make(iou_3d_txt_type_path)
            for i in range(541, 600 + 1):
                cnt += 1
                strip_section_stp_path = f"{data_root}\\strip_section_stp\\type_{j}\\strip_section_{i:0>4}.stp"
                iou_3d_prt_path = f"{iou_3d_prt_type_path}\\iou_3d_{i:0>4}.prt"
                iou_3d_txt_path = f"{iou_3d_txt_type_path}\\iou_3d_{i:0>4}.txt"
                springback_strip_line_path = f"{data_root}\\springback_strip_line\\type_{j}\\springback_strip_line_{i:0>4}.txt"
                prediction_line_path = f"{prediction_line_type_path}\\{i:0>4}.txt"
                sys.stdout = open(iou_3d_txt_path, 'w', encoding = "utf-8")
                no_bias_return = procsssor[f"type_{j}"][0](strip_section_stp_path, iou_3d_prt_path, springback_strip_line_path, prediction_line_path)
                if no_bias_return is not None:
                    iou_3d += no_bias_return
                    global_max_iou_3d = max(global_max_iou_3d, no_bias_return)
                    global_min_iou_3d = min(global_min_iou_3d, no_bias_return)
                else:
                    bias = [0.01, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15]
                    flag = False
                    for b in bias:
                        bias_return = procsssor[f"type_{j}"][1](strip_section_stp_path, iou_3d_prt_path, springback_strip_line_path, prediction_line_path, b)
                        if bias_return is not None:
                            iou_3d += bias_return
                            global_max_iou_3d = max(global_max_iou_3d, bias_return)
                            global_min_iou_3d = min(global_min_iou_3d, bias_return)
                            flag = True
                            break
                    if not flag:
                        error.append(i)
                        cnt -= 1
                sys.stdout = sys.__stdout__
            iou_3d /= cnt
            global_iou_3d += iou_3d
            sys.stdout = open(f"{iou_3d_txt_type_path}\\iou_3d_total.txt", 'w', encoding = "utf-8")
            print(iou_3d)
            print(cnt)
            print(error)
            sys.stdout = sys.__stdout__
        sys.stdout = open(f"{iou_3d_root}\\iou_3d_txt\\global_iou_3d_total.txt", 'w', encoding = "utf-8")
        print(global_iou_3d / 5)
        print(global_max_iou_3d)
        print(global_min_iou_3d)
        sys.stdout = sys.__stdout__