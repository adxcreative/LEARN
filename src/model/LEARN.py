
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder


class PreferenceAlignmentModule(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        summary_config = BertConfig.from_pretrained(
            "bert-base-uncased",
            cache_dir='./checkpoints/',
            local_files_only=True
        )

        if num_layers is not None:
            summary_config.num_hidden_layers = num_layers
        self.summary_encoder = BertEncoder(summary_config)
        text_hidden_size = summary_config.hidden_size
        self.adaptor = nn.Linear(self.input_dim, text_hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(text_hidden_size, self.output_dim),
        )

    @property
    def dtype(self):
        return self.text_encoder.dtype

    @property
    def device(self):
        return self.text_encoder.device

    def get_extended_causal_mask(self, attention_mask):
        bs, seq_length = attention_mask.shape
        seq_ids = torch.arange(seq_length, device=attention_mask.device)
        causal_mask = (
                seq_ids[None, None, :].repeat(bs, seq_length, 1) <= seq_ids[None, :, None]
        )
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = (
                causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        extended_attention_mask = extended_attention_mask.to(attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def extract_user_embedding(self, user_info,
                               history_item_info_list, history_attention_mask):
        bs = len(history_attention_mask)
        history_item_length = len(history_item_info_list)

        last_nonpadding_index = torch.sum(history_attention_mask, dim=1, keepdim=True) - 1
        last_nonpadding_index = torch.tile(last_nonpadding_index.unsqueeze(dim=2), (1, self.output_dim))
        # extended mask
        extended_history_attention_mask = self.get_extended_causal_mask(history_attention_mask)

        history_concat_feat = self.adaptor(history_item_info_list)  # 4096 -> 768
        history_cross_feat = self.summary_encoder(history_concat_feat, attention_mask=extended_history_attention_mask)
        history_cross_feat = self.mlp(history_cross_feat.last_hidden_state)  # bs*(len+1)*256
        # history_cross_feat = self.reduction_dim(history_cross_feat)
        # return last nonpadding features
        user_feat = torch.gather(history_cross_feat, dim=1, index=last_nonpadding_index)  # bs*1*256
        user_feat = user_feat.reshape(bs, self.output_dim)
        user_feat_n = F.normalize(user_feat, dim=-1)
        # return user_feat, user_feat_n
        return user_feat_n

    def extract_item_embedding(self, item_info):
        bs = len(item_info)

        # target
        target_feat = self.adaptor(item_info)  # 4096->768
        target_cross_feat = self.summary_encoder(target_feat)
        target_cross_feat = self.mlp(target_cross_feat.last_hidden_state)

        target_cross_feat = F.normalize(target_cross_feat, dim=-1)  # bs*(1)*256

        # target_cross_feat = F.normalize(item_info, dim=-1)  # bs*(1)*256
        target_cross_feat = target_cross_feat.reshape(bs, self.output_dim)
        return target_cross_feat

    def forward(
            self,
            history_item_info_list,
            history_attention_mask,
            target_item_info_list,
            target_attention_mask,
            item_embedding=None,
    ):
        # extended mask
        extended_history_attention_mask = self.get_extended_causal_mask(history_attention_mask)
        extended_target_attention_mask = self.get_extended_causal_mask(target_attention_mask)

        history_concat_feat = self.adaptor(history_item_info_list)  # 4096 -> 768

        history_cross_feat = self.summary_encoder(
            history_concat_feat, attention_mask=extended_history_attention_mask
        ).last_hidden_state
        history_cross_feat = F.normalize(self.mlp(history_cross_feat), dim=-1)  # bs*len*64

        if item_embedding is None:
            # # target
            target_feat = self.adaptor(target_item_info_list)  # 512->768
            # target_feat = target_item_info_list  # 512->768
            target_cross_feat = self.summary_encoder(
                target_feat, attention_mask=extended_target_attention_mask
            ).last_hidden_state
            target_cross_feat = F.normalize(self.mlp(target_cross_feat), dim=-1)  # bs*(len)*64

            # target_cross_feat = F.normalize(target_item_info_list, dim=-1)

            return history_cross_feat, target_cross_feat
        else:
            return history_cross_feat, item_embedding