from models.base_models import SingleFrame, EarlyFusion, LateFusion, CNNLSTM, S3D, ViViT
from models.mutimodal_models import MultiModalViViT, MultiModalS3D, MultiModalCNNLSTM, MultiModalEarlyFusion, MultiModalLateFusion
from models.attention_multimodal_models import AttentionMultiModalViViT, AttentionMultiModalS3D


def get_model(model_name, num_classes, num_frames=None):
    if model_name == 'single_frame':
        return SingleFrame(num_classes)
    elif model_name == 'early_fusion':
        return EarlyFusion(num_classes, num_input_channels=num_frames)
    elif model_name == 'late_fusion':
        return LateFusion(num_classes)
    elif model_name == 'cnn_lstm':
        return CNNLSTM(num_classes)
    elif model_name == 's3d':
        return S3D(num_classes)
    elif model_name == 'vivit':
        return ViViT(num_classes)
    elif model_name == 'multimodal_vivit':
        return MultiModalViViT(num_classes)
    elif model_name == 'multimodal_s3d':
        return MultiModalS3D(num_classes)
    elif model_name == 'multimodal_cnn_lstm':
        return MultiModalCNNLSTM(num_classes)
    elif model_name == 'multimodal_late_fusion':
        return MultiModalLateFusion(num_classes)
    elif model_name == 'multimodal_early_fusion':
        return MultiModalEarlyFusion(num_classes)
    elif model_name == 'attention_multimodal_vivit':
        return AttentionMultiModalViViT(num_classes)
    elif model_name == 'attention_multimodal_s3d':
        return AttentionMultiModalS3D(num_classes)
