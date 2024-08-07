### etri_autocalib_transformer_ViT_Model

Files are copied from https://github.com/jeonsworld/ViT-pytorch

Modifications:

1. **Inputs**
    - The model is still only tested on KITTI360 image (`rgb_inputs`) mimicking with `torch.randn([1, 3, 1400, 1400])` shape.
    - The depth model (`depth_inputs`) is created by modifying `rgb_inputs` into grayscale and rotating 180.

2. **Current Neural Network Architecture**
    - Created two embeddings for each input (`rgb_embedding_output` & `depth_embedding_output`).
    - Transformer patch size: (100, 100).
    - Both attentions are concatenated.
    - FC layer added after concatenation.
    - Head is added (simple Linear layer).
    - Adjusted for Dummy DataLoader (included).

3. **Dummy DataLoader**
    - DataLoader is created based on the KITTI-360 images (1GB only).
    - DataLoader output format: ([image_rgb, image_depth], labels).
    - batch size: 20

4. **Dummy training**
    - Training is started by preprocessing dataloader.
    - MSE Loss is used on the training.
