import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpi4py import MPI
import time
import sys

low_threshold, high_threshold = 150, 240

def img2gray(img_path):
    img = np.asarray(Image.open(img_path))
    gray_img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    return gray_img

def cal_2d_distance(x, y, flag = "Euclidean"):
    if flag == "Euclidean":
        return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
    elif flag == "Manhattan":
        return np.abs(x[0] - x[1]) + np.abs(y[0] - y[1])
    else:
        raise Exception("Not support this type of distance.")

def gray2sdf(offset, gray_image, boundary):

    h, w = gray_image.shape
    
    sdf = np.zeros((h, w), dtype = float)

    for i in range(h):
        for j in range(w):
            if gray_image[i][j] < low_threshold: continue
            min_distance = 1e10
            for b in boundary:
                min_distance = min(min_distance, cal_2d_distance(b, [i + offset, j]))
            sdf[i][j] = min_distance
            if gray_image[i][j] > high_threshold: sdf[i][j] = -sdf[i][j]
    return sdf

def get_boundary(gray_image):
    h, w = gray_image.shape
    boundary = []
    for i in range(h):
        for j in range(w):
            if gray_image[i][j] < low_threshold:
                boundary.append([i, j])
    return boundary

def main(read_path, save_path):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        start_time = time.time()
        gray_image = img2gray(read_path)
        boundary = np.array(get_boundary(gray_image), dtype = float)
        b_len = len(boundary)
        comm.Bcast(np.array(b_len, dtype = int), root = 0)
    else:
        b_len = np.empty(1, dtype = int)
        comm.Bcast(b_len, root = 0)
        boundary = np.empty((b_len[0], 2), dtype = float)

    comm.Bcast(boundary, root = 0)

    if rank == 0:
        h, w = gray_image.shape
        sdf = np.zeros((h, w), dtype = float)
        block = h // size
        remainder = h % size
        start, end = [], []

        requests = []

        for r in range(size):
            if r < remainder:
                start.append(r * (block + 1))
                end.append(start[-1] + block + 1)
            else:
                start.append(r * block + remainder)
                end.append(start[-1] + block)
            if r != 0:
                comm.Send(np.array([start[-1], end[-1] - start[-1], w], dtype = int), dest = r)
                requests.append(comm.Isend(gray_image[start[-1] : end[-1], :], dest = r))
        sdf[start[0] : end[0], :] = gray2sdf(start[0], gray_image[start[0] : end[0], :], boundary)
        MPI.Request.Waitall(requests)
        for r in range(1, size):
            comm.Recv(sdf[start[r] : end[r], :], source = r)
        end_time = time.time()
        print(f"Total elapsed time: {end_time - start_time} seconds")
        sys.stdout.flush()
        np.savetxt(save_path, sdf)
        
    else:
        recv_shape = np.empty(3, dtype = int)
        comm.Recv(recv_shape, source = 0)
        recv_data = np.empty(recv_shape[1:], dtype = np.uint8)
        req = comm.Irecv(recv_data, source = 0)
        req.Wait()
        comm.Send(gray2sdf(recv_shape[0], recv_data, boundary), dest = 0)

    comm.Barrier()

if  __name__ == "__main__":
    for i in range(3, 4):
        for j in range(27, 400 + 1):
            print(f"processing type_{i} {j:0>4}.jpg...")
            sys.stdout.flush()
            read_path = rf"./strip_section_jpg/type_{i}/{j:0>4}.jpg"
            save_path = rf"./strip_section_sdf/type_{i}/{j:0>4}.txt"
            main(read_path, save_path)
    for i in range(4, 5):
        for j in range(1, 400 + 1):
            print(f"processing type_{i} {j:0>4}.jpg...")
            sys.stdout.flush()
            read_path = rf"./strip_section_jpg/type_{i}/{j:0>4}.jpg"
            save_path = rf"./strip_section_sdf/type_{i}/{j:0>4}.txt"
            main(read_path, save_path)