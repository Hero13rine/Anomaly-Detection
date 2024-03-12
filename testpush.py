#this is a file that test push github from other computer
import tensorflow as tf
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))