# shellcheck disable=SC1101
python projects/UniDet/demo/demo.py --config-file projects/UniDet/configs/Unified_learned_OCIM_R50_6x+2x.yaml \
--input images=./tmp/*.jpg \
--output=./tmp_data \
--opts MODEL.WEIGHTS models/Unified_learned_OCIM_R50_6x+2x.pth