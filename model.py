from transformers import AutoModel
from transformers import RobertaConfig
from bert_model import BertCrossLayer, BertAttention
from clip_model import build_model
import torch
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# Co-attention module:
class ViVQANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_emb_size = cfg['hidden_size']
        self.image_emb_size = cfg['hidden_size']
        self.hidden_size = cfg['hidden_size']
        self.num_class = cfg['model_params']['num_class']
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_config = RobertaConfig(
                vocab_size=cfg['model_params']['coattn']["vocab_size"],
                hidden_size=cfg['model_params']["hidden_size"],
                num_hidden_layers=cfg['model_params']['coattn']["num_layers"],
                num_attention_heads=cfg['model_params']['coattn']["num_heads"],
                intermediate_size=cfg['model_params']["hidden_size"] * cfg['model_params']['coattn']["mlp_ratio"],
                max_position_embeddings=cfg['model_params']['coattn']["max_text_len"],
                hidden_dropout_prob=cfg['model_params']['coattn']["drop_rate"],
                attention_probs_dropout_prob=cfg['model_params']['coattn']["drop_rate"],
            )
        
        # Load ViT and PhoBERT:
        self.text_transformer = AutoModel.from_pretrained(cfg['model_params']['text_encoder']['pretrained_model'])
        self.vit_model = build_model(cfg['model_params']['image_encoder']['vit'], resolution_after=224)

        # Coattention Module:

        self.token_type_embeddings = nn.Embedding(2, cfg['hidden_size'])
        self.token_type_embeddings.apply(init_weights)

        self.cross_modal_text_transform = nn.Linear(self.text_emb_size, self.hidden_size)
        self.cross_modal_text_transform.apply(init_weights)
        self.cross_modal_image_transform = nn.Linear(self.image_emb_size, self.hidden_size)
        self.cross_modal_image_transform.apply(init_weights)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(self.bert_config) for _ in range(6)])
        self.cross_modal_image_layers.apply(init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(self.bert_config) for _ in range(6)])
        self.cross_modal_text_layers.apply(init_weights)

        self.cross_modal_image_pooler = Pooler(self.hidden_size)
        self.cross_modal_image_pooler.apply(init_weights)
        self.cross_modal_text_pooler = Pooler(self.hidden_size)
        self.cross_modal_text_pooler.apply(init_weights)

        # Init classifier:
        self.vqa_classifier = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.num_class),
            )
        self.vqa_classifier.apply(init_weights)

    def forward(self, batch):
        text = torch.squeeze(batch['input_ids'], 1).to(self.device)
        att_mask = torch.squeeze(batch['attention_mask'], 1).to(self.device)
        image = batch['image_tensor'].to(self.device)
        label = batch['label'].to(self.device)


        input_shape = att_mask.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(att_mask, input_shape, self.device)
        text_embeds = self.text_transformer(text, att_mask)[0]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        image_embeds = self.vit_model(image)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=self.device)
        extend_image_masks = self.text_transformer.get_extended_attention_mask(image_masks, image_masks.size(), self.device)

        # text_embeds, image_embeds = (
        #     text_embeds + self.token_type_embeddings(torch.zeros_like(att_mask)),
        #     image_embeds
        #     + self.token_type_embeddings(
        #         torch.full_like(torch.zeros_like(image_masks))
        #     ),
        # )

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        cls_feats_image = self.cross_modal_image_pooler(y)
        # avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        # cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats_text = self.cross_modal_text_pooler(x)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        logits = self.vqa_classifier(cls_feats)
        return logits