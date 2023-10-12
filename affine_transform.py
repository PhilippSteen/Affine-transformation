# Luciano A. Masullo, Philipp R. Steen
# Jungmann Lab
# Max Planck Institute of Biochemistry, 2023

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import h5py

def matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    v0 = np.array(np.transpose(v0), dtype=np.float64, copy=True)
    v1 = np.array(np.transpose(v1), dtype=np.float64, copy=True)
    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")
    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)
    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)
    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= np.sqrt(np.sum(v1) / np.sum(v0))
    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M
_EPS = np.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [0.0,                 0.0,                 0.0, 1.0]])

def vector_norm(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)

def Import(path):
    fulltable = pd.read_hdf(path, key = 'locs')
    return(fulltable)

def GroupCenter(data):
    #Calculates the center of each group to make a "RESI" localization
    xvals = []
    yvals = []
    for i in range(0, max(data['group'])):
        xvals.append(data.loc[data['group'] == i, 'x'].mean())
        yvals.append(data.loc[data['group'] == i, 'y'].mean())
    return(xvals, yvals)

def SimplePlot2(subject,reference):
    fig,ax = plt.subplots(1, figsize = (10,10))
    ax.scatter(subject[0], subject[1], color = "blue", label = "subject")
    ax.scatter(reference[0], reference[1], color = "red", label = "reference")
    ax.axis('equal')
    ax.legend()
    plt.show()

def Rectifier(v):
    v = np.array(np.transpose(v), dtype=np.float64, copy=True)
    return(v)

def apply_affine_transformation_to_coordinates(transformation_matrix, coordinates):
    # Convert the coordinates to homogeneous coordinates
    homogeneous_coordinates = np.hstack((coordinates, np.ones((len(coordinates), 1))))
    # Apply the affine transformation
    transformed_coordinates = np.matmul(homogeneous_coordinates, transformation_matrix.T)
    # Normalize the transformed coordinates
    transformed_coordinates = transformed_coordinates[:, :2] / transformed_coordinates[:, 2:]
    return(transformed_coordinates)

def picasso_hdf5(df, hdf5_fname, hdf5_oldname, path):
    labels = list(df.keys())
    df_picasso = df.reindex(columns=labels, fill_value=1)
    locs = df_picasso.to_records(index = False)
    # Saving data
    hf = h5py.File(path + hdf5_fname, 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()
    # YAML saver
    yaml_oldname = path + hdf5_oldname.replace('.hdf5', '.yaml')
    yaml_newname = path + hdf5_fname.replace('.hdf5', '.yaml')
    yaml_file_info = open(yaml_oldname, 'r')
    yaml_file_data = yaml_file_info.read()
    yaml_newfile = open(yaml_newname, 'w')
    yaml_newfile.write(yaml_file_data)
    yaml_newfile.close()   
    print('New Picasso-compatible .hdf5 file and .yaml file successfully created.')

def ApplyToData(path, file, matrix, savename, save): #This allows you to apply the transformation to your data
    data = Import(path+file)
    coords = data[["x", "y"]].values
    transformed = apply_affine_transformation_to_coordinates(matrix, coords)
    newdata = data.copy()
    newdata[["x","y"]] = transformed
    if save:
        picasso_hdf5(newdata, savename+"_corr.hdf5", file, path)
    
def FindMatrix(subject, reference, plot):
    mean_s_x = []
    mean_s_y = []
    mean_r_x = []
    mean_r_y = []
    for index, entry in enumerate(subject):
        data_s = Import(subject[index])
        data_r = Import(reference[index])
        smean_s = GroupCenter(data_s)
        smean_r = GroupCenter(data_r)
        mean_s_x.append(smean_s[0])
        mean_s_y.append(smean_s[1])
        mean_r_x.append(smean_r[0])
        mean_r_y.append(smean_r[1])
    mean_s_x = np.concatenate(mean_s_x, axis=0)
    mean_s_y = np.concatenate(mean_s_y, axis=0)
    mean_r_x = np.concatenate(mean_r_x, axis=0)
    mean_r_y = np.concatenate(mean_r_y, axis=0)
    mean_s = [mean_s_x, mean_s_y]
    mean_r = [mean_r_x, mean_r_y]
    if plot:
        SimplePlot2(mean_s,mean_r)
    new_s = Rectifier(mean_s)
    new_r = Rectifier(mean_r)
    matrix = matrix_from_points(new_s, new_r)
    corrected_subject = apply_affine_transformation_to_coordinates(matrix, new_s)
    #Find approximate accuracy
    delta_x = abs(corrected_subject[:,0] - new_r[:,0])**2
    delta_y = abs(corrected_subject[:,1] - new_r[:,1])**2
    delta_squared = delta_x + delta_y
    delta = np.sqrt(delta_squared)*130
    print("There were {} beads".format(len(delta)))
    print("The approximate accuracy is {:.2f} nm".format(np.mean(delta)))
    if plot:
        SimplePlot2(corrected_subject.T, mean_r)
    return(matrix)


############### User input below

all_green = ["path to your first FOV",
            "path to your second FOV",
            "...",
            "path to your n-th FOV"]

all_red = ["path to your first FOV",
            "path to your second FOV",
            "...",
            "path to your n-th FOV"]

red_to_green = FindMatrix(all_red, all_green, plot = True)

ApplyToData("file path", "file name of the data to be transformed", red_to_green, "new file name", save=True)




