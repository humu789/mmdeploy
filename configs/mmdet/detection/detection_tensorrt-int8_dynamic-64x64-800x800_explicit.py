_base_ = [
    '../_base_/base_dynamic.py', '../../_base_/backends/tensorrt.py'
]

backend_config = dict(
    common_config=dict(
        max_workspace_size=1 << 30,
        int8_mode=True,
        explicit_quant_mode=True),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 64, 64],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 800, 800])))
    ])

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

quantizer=dict(
    type='mmrazor.TensorRTQuantizer',
    global_qconfig=global_qconfig,
    tracer=dict(
        type='mmrazor.CustomTracer',
        skipped_methods=[
            'mmdet.models.dense_heads.base_dense_head.BaseDenseHead.predict_by_feat',  # noqa: E501
            'mmdet.models.dense_heads.anchor_head.AnchorHead.loss_by_feat',
        ]))

checkpoint='/nvme/humu/experiments/retina_trt/model_ptq_deploy.pth'