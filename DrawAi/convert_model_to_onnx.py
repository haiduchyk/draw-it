import tensorflow as tf
from tensorflow.python.saved_model import builder as pb_builder
import tf2onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic


pre_model = tf.keras.models.load_model('models/QuickDraw.h5')
pre_model.summary()

onnx_model, _ = tf2onnx.convert.from_keras(model=pre_model)
tf2onnx.onnx.save_model(onnx_model, 'model.onnx')