import argparse
import numpy as np
import cv2

def hm_sum(hml, hmr):
    #return cv2.addWeighted(hml, 0.5, hmr, 0.5, 0)
    return cv2.add(hml, hmr)


def main():
    parser = argparse.ArgumentParser(description="Load two numpy arrays from files")
    parser.add_argument("file1", type=str, help="Path to first raw heatmap")
    parser.add_argument("file2", type=str, help="Path to second raw heatmap")
    parser.add_argument("file3", type=str, help="Path to original image (for overlay)")
    
    args = parser.parse_args()
    
    hm1 = cv2.imread(args.file1)
    hm2 = cv2.imread(args.file2)
    raw_img = cv2.imread(args.file3)
    h, w, c = raw_img.shape
    
    print(f"Heatmap 1 shape: {hm1.shape}")
    print(f"Heatmap 2 shape: {hm2.shape}")
    print(f"Original image shape: {raw_img.shape}")

    res = hm_sum(hm1, hm2)

    out_img = cv2.applyColorMap(res, cv2.COLORMAP_JET)
    out_img = cv2.resize(out_img, (w, h))

    out_img = out_img * 0.5 + raw_img * 0.5

    cv2.imwrite("merged_hm.png", out_img)



if __name__ == "__main__":
    main()
