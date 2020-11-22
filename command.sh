# docker run -u $(id -u):$(id -g) -v ~/git/test_task:/usr/src --privileged=true  -w /usr/src edwina1030/pmb_ling:v5 python3 dataset_split.py
docker run -it -u $(id -u):$(id -g) --name test_container -v /home/WIN-UNI-DUE/sotiling:/sotiling edwina1030/pmb_ling:v5
