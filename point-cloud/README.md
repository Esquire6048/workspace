# Commands

python modelnet40_slice.py \
--file_list dataset/modelnet40_ply_hdf5_2048/train_files.txt \
--sample_idxs 0 10 42 \
--out_dir cache/output_ply \
--slices 3 \
--fov_deg 120