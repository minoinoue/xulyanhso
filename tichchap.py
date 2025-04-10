import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(r"D:\Uni\XLA\Thuc_Hanh\Bai2\image2.png", flags=0)

# Ma trận mặt nạ 1
mask1 = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.float32) / 5

# Ma trận mặt nạ 2
mask2 = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]], dtype=np.float32) / 9

# Ma trận mặt nạ 3
mask3 = np.array([[1, 3, 1],
                  [3, 16, 3],
                  [1, 3, 1]], dtype=np.float32) / 32

# Ma trận mặt nạ 4
mask4 = np.array([[0, 1, 0],
                  [1, 4, 1],
                  [0, 1, 0]], dtype=np.float32) / 8

def loc_tb(img, mask):
    pad_h = mask.shape[0] // 2
    pad_w = mask.shape[1] // 2
    padded_img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    output_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            roi = padded_img[i:i + mask.shape[0], j:j + mask.shape[1]]
            output_img[i, j] = np.sum(roi * mask)
    return output_img.astype(np.uint8)


if img is None:
    print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
else:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')

    img_loc_mask1 = loc_tb(img.astype(np.float32), mask1)

    plt.subplot(1, 2, 2)
    plt.imshow(img_loc_mask1, cmap='gray')
    plt.title('Ảnh sau lọc với mặt nạ 1')
    plt.axis('off')
    plt.tight_layout()
    plt.show()