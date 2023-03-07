_base_ = ['../_base_/base_openvino_dynamic-800x1344.py']

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

# TODO save in mmrazor's checkpoint
quantizer=dict(
    type='mmrazor.OpenVINOQuantizer',
    global_qconfig=global_qconfig,
    tracer=dict(
        type='mmrazor.CustomTracer',
        skipped_methods=[
            'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
            'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',
        ]))

checkpoint='/mnt/petrelfs/humu/experiments/yolox-s_openvino/model_ptq_deploy.pth'
