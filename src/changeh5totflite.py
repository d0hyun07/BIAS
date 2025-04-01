import tensorflow as tf

model = tf.keras.models.load_model("C:/Users/u0102/Desktop/BIAS/models/best_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 핵심 설정 추가!!
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # 기본 TFLite 연산
    tf.lite.OpsSet.SELECT_TF_OPS          # TensorFlow 연산 일부 사용 허용
]

# TensorList 오류 해결 옵션
converter._experimental_lower_tensor_list_ops = False

# (선택) 양자화 적용
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 변환 실행
tflite_model = converter.convert()

# 저장
with open("bias_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ 모델 변환 완료!")
