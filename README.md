# distributed_training_using_tensorflow

This directory shows the distributed training strategy , where DTS is used to train larger Models which cannot be handled using single GPU.






![parameter-server-strategy](https://user-images.githubusercontent.com/37244966/190893502-6fb72f0c-45dd-43bf-a094-91c39a3ef5af.png)















In TensorFlow, the TF_CONFIG environment variable is required for training on multiple machines, each of which possibly has a different role. TF_CONFIG is a JSON string used to specify the cluster configuration on each worker that is part of the cluster.

There are two components of TF_CONFIG: cluster and task.

Let's dive into how they are used:

cluster:

It is the same for all workers and provides information about the training cluster, which is a dict consisting of different types of jobs such as worker.

In multi-worker training with MultiWorkerMirroredStrategy, there is usually one worker that takes on a little more responsibility like saving checkpoint and writing summary file for TensorBoard in addition to what a regular worker does.

Such a worker is referred to as the chief worker, and it is customary that the worker with index 0 is appointed as the chief worker (in fact this is how tf.distribute.Strategy is implemented). Ref ( MLOPs specialization coursera)




task:

Provides information of the current task and is different on each worker. It specifies the type and index of that worker.





Steps to perform Multiworker training 
1-Create your dataloading/Model script and call it MNIST.py
-follow ETL rules ( Extract , transform and load ) your data in case it needs data preprocessing and cleaning 
-Build simple model using higher level Tensorflow APis such as keras
-Define your metrics : loss function-optimizer





2-load the model in the distributed_training.py/.ipynb source file by 

import mnist


3-Configure both the Cluster & task 
-In multi-worker training with MultiWorkerMirroredStrategy, there is usually one worker that takes on a little more responsibility like saving checkpoint and writing summary file for TensorBoard in addition to what a regular worker does.

-Such a worker is referred to as the chief worker, and it is customary that the worker with index 0 is appointed as the chief worker (in fact this is how tf.distribute.Strategy is implemented).



Task Provides information of the current task and is different on each worker. It specifies the type and index of that worker.

in distributed_training.py

tf_config = {
    'cluster': {
        'worker': ['external_machine_ip1', 'external_machine_ip2']
    },
    'task': {'type': 'worker', 'index': 0}
}



4-Creating training script and call it main.py
- call mnist.py by import mnist
- Get TF_CONFIG from the env variables and save it as JSON
-Infer number of workers from tf_config
-Define strategy
-Define global batch size
-Load dataset
-Create and compile model following the distributed strategy
-Train the model

5-launching workers refered to job assigned 
-%%bash --bg
python main.py &> job_0.log


for first worker set the index to 0
tf_config['task']['index'] = 0
os.environ['TF_CONFIG'] = json.dumps(tf_config)


for second worker set the index to 1

tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)





****************************************************************************************************************************************************

Some notes i gained from experience and practice 

1-Hardware is better to have same identical features/synchronous clocks 
from experience 
-if you want train a model on 2 Gpus on single machine , then its better to have the same 2 GPU
ie GTX1080 with GTX1080 


-on different devices , the devices must be on the same network ,better to have same clock speed (very close processor frequencies, same GPU specs) to avoid idle mood or lagging in the shake hand stage 


-use check points 

