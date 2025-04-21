"""
Text-Image Retrieval のモデル定義
"""

import copy
import math

import torch
import torch.nn.functional as F

import clip


class ClipReverie(torch.nn.Module):
    """
    Baseline : CLIP ViT-L/14 を用いて評価
    """

    def __init__(self, clip_base, device, bbox=True, N=30, with_chat_gpt_=True):
        super(ClipReverie, self).__init__()
        self.clip_model, self.preprocess_clip = clip.load(clip_base, device=device)
        for params in self.clip_model.parameters():
            params.requires_grad = False
        self.with_chat_gpt = with_chat_gpt_
        self.temperature = 1
        self.bbox = bbox
        self.N = N

        self.encoder_np_txt = self.create_encoder(8, 768, 768, 0.3, 4)
        self.encoder_img = self.create_encoder(8, 768 * 3, 768, 0.3, 3)

        self.relu = torch.nn.ReLU()

        self.fc30 = torch.nn.Linear(1536, 1000)
        self.fc31 = torch.nn.Linear(1000, 768)
        # ID 104
        if self.with_chat_gpt:
            self.fc32 = torch.nn.Linear(768 * 5, 768 * 2)
        else:
            self.fc32 = torch.nn.Linear(768 * 2, 768 * 4)  # ChatGPT由来のベクトルを無視するため
        self.bn30 = torch.nn.BatchNorm1d(1000)
        self.dropout30 = torch.nn.Dropout(0.4)
        self.fc33 = torch.nn.Linear(768 * 2, 768)
        self.bn32 = torch.nn.BatchNorm1d(768 * 2)
        self.dropout32 = torch.nn.Dropout(0.3)

        self.fc60 = torch.nn.Linear(4096, 768)

        self.fc90 = torch.nn.Linear(768 * 8, 768 * 5)
        self.fc91 = torch.nn.Linear(768 * 5, 768 * 3)
        self.dropout90 = torch.nn.Dropout(0.3)
        self.bn90 = torch.nn.LayerNorm(768 * 5)

        # llava * vit * clip
        self.fc100 = torch.nn.Linear(768 * 3, 768 * 4)
        self.fc101 = torch.nn.Linear(768 * 4, 768 * 6)
        self.fc102 = torch.nn.Linear(768 * 6, 768 * 6)
        self.fc103 = torch.nn.Linear(768 * 6, 768 * 4)
        self.fc104 = torch.nn.Linear(768 * 4, 768 * 2)
        self.fc105 = torch.nn.Linear(768 * 2, 768)
        self.bn100 = torch.nn.LayerNorm(768 * 4)
        self.bn101 = torch.nn.LayerNorm(768 * 6)
        self.bn102 = torch.nn.LayerNorm(768 * 6)
        self.bn103 = torch.nn.LayerNorm(768 * 4)
        self.bn104 = torch.nn.LayerNorm(768 * 2)
        self.dropout100 = torch.nn.Dropout(0.3)
        self.dropout101 = torch.nn.Dropout(0.3)
        self.dropout102 = torch.nn.Dropout(0.3)
        self.dropout103 = torch.nn.Dropout(0.3)
        self.dropout104 = torch.nn.Dropout(0.3)

        # self.fc110 = torch.nn.Linear(768 * 8, 768 * 2)

        self.fc111 = torch.nn.Linear(768 * 8, 768)
        self.ln111 = torch.nn.LayerNorm(768)
        self.dropout111 = torch.nn.Dropout(0.3)

    def image_encoder(self, image, llava_feature, gpt4v_embeddings):
        llava_feature = llava_feature.float()
        gpt4v_embeddings = gpt4v_embeddings.float()  # [bs, 768 * 4]

        llava_feature = self.fc60(llava_feature.squeeze(1))

        image_original = image[:, 0, :]  # torch.Size([bs, 768])
        image_sam = image[:, 1, :]  # torch.Size([bs, 768])
        image_vit = image[:, 2, :]  # torch.Size([bs, 768])
        image_embeddings = torch.cat(
            [
                image_vit.float(),
                image_sam.float(),
                image_original.float(),
                llava_feature.float(),
                gpt4v_embeddings.float(),
            ],
            dim=1,
        )

        identify = image_embeddings

        image_embeddings = self.relu(self.fc90(image_embeddings))
        image_embeddings = self.dropout90(self.bn90(image_embeddings))
        image_embeddings = self.fc91(image_embeddings)

        image_embeddings = self.encoder_img(image_embeddings)[:, 0, :].squeeze(1)

        image_embeddings = self.relu(self.fc100(image_embeddings))
        image_embeddings = self.dropout100(self.bn100(image_embeddings))
        image_embeddings = self.relu(self.fc101(image_embeddings))
        image_embeddings = self.dropout101(self.bn101(image_embeddings))
        image_embeddings = self.relu(self.fc102(image_embeddings))
        image_embeddings = self.dropout102(self.bn102(image_embeddings))
        image_embeddings = self.relu(self.fc103(image_embeddings))
        image_embeddings = self.dropout103(self.bn103(image_embeddings))
        image_embeddings = self.relu(self.fc104(image_embeddings))
        image_embeddings = self.dropout104(self.bn104(image_embeddings))
        image_embeddings = self.fc105(image_embeddings)
        image_embeddings += self.dropout111(self.ln111(self.fc111(identify)))

        return image_embeddings.half()

    def text_encoder(
        self,
        tokenized_text_clip,
        tokenized_instruction_modified_clip,
        tokenized_text_chatgpt_clip,
        tokenized_np_clip,
        gpt3_embeddings=None,
    ):
        text_embeddings = self.clip_model.encode_text(tokenized_text_clip).float()

        if self.with_chat_gpt:
            modified_text_embeddings = self.clip_model.encode_text(tokenized_instruction_modified_clip).float()
            text_chatgpt_clip = self.clip_model.encode_text(tokenized_text_chatgpt_clip).float().unsqueeze(1)
            n_np_embeddings = torch.cat(
                [text_chatgpt_clip, self.clip_model.encode_text(tokenized_np_clip[:, 0, :]).float().unsqueeze(1)], dim=1
            )

            for n in range(1, self.N):
                n_np_embeddings = torch.cat(
                    [
                        n_np_embeddings,
                        self.clip_model.encode_text(tokenized_np_clip[:, n, :]).float().unsqueeze(1),
                    ],
                    dim=1,
                )

            # GPT3
            gpt3_embeddings = self.relu(self.fc30(gpt3_embeddings))
            gpt3_embeddings = self.fc31(self.dropout30(self.bn30(gpt3_embeddings)))

            np_embeddings = self.encoder_np_txt(n_np_embeddings)[:, 0, :].squeeze(1)  # [bs, 768]

            embeddings = torch.cat(
                [
                    text_chatgpt_clip.squeeze(1),
                    text_embeddings,
                    modified_text_embeddings,
                    np_embeddings,
                    gpt3_embeddings,
                ],
                dim=1,
            )
        else:
            # tokenized_text_clip, tokenized_np_clipのみを利用(つまり、ChatGPTは利用しない)して、embeddingsを生成
            embeddings = torch.cat(
                [
                    text_embeddings,
                    self.clip_model.encode_text(tokenized_np_clip[:, 0, :]).float(),
                ],
                dim=1,
            )

        x = self.relu(self.fc32(embeddings))
        x = self.dropout32(self.bn32(x))
        text_embeddings = self.fc33(x)

        return text_embeddings.half()

    def calc_logits(
        self,
        image,
        text_clip,
        tokenized_instruction_modified_clip,
        text_chatgpt_clip,
        tokenized_np_clip,
        gpt3_embeddings,
        llava_feature,
        gpt4v_embeddings,
        _eval=False,
    ):
        image_embeddings = self.image_encoder(image, llava_feature, gpt4v_embeddings)
        text_embeddings = self.text_encoder(
            text_clip,
            tokenized_instruction_modified_clip,
            text_chatgpt_clip,
            tokenized_np_clip,
            gpt3_embeddings,
        )

        logits = self.calc_logits_from_embeddings(image_embeddings, text_embeddings)

        return logits, image_embeddings, text_embeddings

    def calc_logits_from_embeddings(self, image_embeddings, text_embeddings):
        # [128,128]
        return (text_embeddings @ image_embeddings.T) / self.temperature

    def forward(
        self,
        entire_image_feature,
        tokenized_instruction_clip,
        tokenized_instruction_modified_clip,
        tokenized_instruction_chatgpt_clip,
        tokenized_np_clip,
        gpt3_embeddings,
        llava_feature,
        gpt4v_embeddings,
    ):
        logits, image_embeddings, text_embeddings = self.calc_logits(
            entire_image_feature,
            tokenized_instruction_clip,
            tokenized_instruction_modified_clip,
            tokenized_instruction_chatgpt_clip,
            tokenized_np_clip,
            gpt3_embeddings,
            llava_feature,
            gpt4v_embeddings,
        )

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax((images_similarity + texts_similarity) / 2.0 * self.temperature, dim=-1)
        return logits, targets, image_embeddings, text_embeddings

    def preprocess(self, x):
        return self.preprocess_clip(x)

    def create_encoder(self, h, d_model, d_ff, dropout, N):
        config = TransformerConfig(h, d_model, d_ff, dropout, N)
        return config.create_encoder()


class Encoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(torch.nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.

    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.parameter.Parameter(torch.ones(features))
        self.b_2 = torch.nn.parameter.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """Definition of forward propagation"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """Definition of forward propagation"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    """Take in model size and number of heads."""

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implement Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l_fn(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l_fn, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# class CrossAttentionEncoder(torch.nn.Module):
#     """An encoder stack that applies cross-attention across all its layers."""
#
#     def __init__(self, N, layer):
#         super(CrossAttentionEncoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, query, kv):
#         """Pass the query and kv through each layer in turn."""
#         for layer in self.layers:
#             query = layer(query, kv)
#         return self.norm(query)
#
#
# class CrossAttentionEncoderLayer(torch.nn.Module):
#     """Encoder is made up of self-attn and feed forward (defined below)"""
#
#     def __init__(self, size, cross_attn, feed_forward, dropout):
#         super(CrossAttentionEncoderLayer, self).__init__()
#         self.cross_attn = cross_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size
#
#     def forward(self, x, kv):
#         """Follow Figure 1 (left) for connections."""
#         x = self.sublayer[0](x, lambda x: self.cross_attn(x, kv, kv))
#         return self.sublayer[1](x, self.feed_forward)


class CrossAttentionEncoder(torch.nn.Module):
    """An encoder stack that applies cross-attention across all its layers."""

    def __init__(self, N, layer):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, query, kv):
        """Pass the query and kv through each layer in turn."""
        for layer in self.layers:
            query = layer(query, kv)
        return self.norm(query)


class CrossAttentionEncoderLayer(torch.nn.Module):
    """Encoder is made up of cross-attn and feed forward (defined below)"""

    def __init__(self, size, cross_attn, feed_forward, dropout):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, cross_inputs):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, cross_inputs, cross_inputs))
        return self.sublayer[1](x, self.feed_forward)


class CrossAttention(torch.nn.Module):
    """Take in model size and number of heads."""

    def __init__(self, h, d_model, dropout=0.1):
        super(CrossAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implement Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l_fn(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l_fn, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CrossTransformerConfig:
    """Transformer settings"""

    def __init__(self, h, d_model, d_ff, dropout, N):
        self.h = h
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N

    def create_encoder(self):
        c = copy.deepcopy
        cross_attn = CrossAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        return CrossAttentionEncoder(
            self.N, CrossAttentionEncoderLayer(self.d_model, c(cross_attn), c(ff), dropout=self.dropout)
        )


class TransformerConfig:
    """Transformer settings"""

    def __init__(self, h, d_model, d_ff, dropout, N):
        self.h = h
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N

    def create_encoder(self):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        return Encoder(self.N, EncoderLayer(self.d_model, c(attn), c(ff), dropout=self.dropout))


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
