
cd ../mains/

:: this is to run train,you can change data_steps for diferrent steps
python original_data.py --dataset_dir ../metr/ --data_steps 3-3
python RMASTGRU.py --dataset_dir ../metr/ --data_steps 3-3 --features 13 --data 1 --experiment  1 --test 0  --Times 1 --GPU 2 --start_time 0 --time_slot 5 --Batch_Size 16 --T  288 --round 0 --graph_pkl adj_mx_metr.pkl

:: this is to run test,you can change data_steps for diferrent steps
::python original_data.py --dataset_dir ../metr/ --data_steps 3-3
::python RMASTGRU.py --dataset_dir ../metr/ --data_steps 3-3 --features 13 --data 1 --experiment  0 --test 1  --Times 1 --GPU 2 --start_time 0 --time_slot 5 --Batch_Size 16 --T  288 --round 0 --graph_pkl adj_mx_metr.pkl

:: this is to run train,you can change data_steps for diferrent steps
::python original_data.py --dataset_dir ../pems/ --data_steps 3-3
::python RMASTGRU.py --dataset_dir ../pems/ --data_steps 3-3 --features 13 --data 1 --experiment  1 --test 0  --Times 1 --GPU 2 --start_time 0 --time_slot 5 --Batch_Size 16 --T  288 --round 0 --graph_pkl adj_mx_bay.pkl

:: this is to run test,you can change data_steps for diferrent steps
::python original_data.py --dataset_dir ../pems/ --data_steps 3-3
::python RMASTGRU.py --dataset_dir ../pems/ --data_steps 3-3 --features 13 --data 1 --experiment  0 --test 1  --Times 1 --GPU 2 --start_time 0 --time_slot 5 --Batch_Size 16 --T  288 --round 0 --graph_pkl adj_mx_bay.pkl


cd ../bat/
