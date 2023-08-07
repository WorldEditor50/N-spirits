#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP
#include <assert.h>
#include <tuple>
#include <vector>
#include "activate.hpp"
#include "../dl/layer.hpp"
#include "../basic/tensor.hpp"
#include "../basic/util.hpp"


#if 0

class SelfAttention
{
public:
    int embedSize;
    int headDim;
    int headNum;
    FcLayer values;
    FcLayer keys;
    FcLayer queries;
    FcLayer fc_out;
public:
    SelfAttention(){}
    SelfAttention(int embedSize_, int headNum_)
    {
        embedSize = embedSize_;
        headNum = headNum_;
        headDim = embedSize;
        /* "Embed size needs to be div by heads"  */
        assert(headDim*headNum == embedSize);

        values  = FcLayer(headDim, headDim, false, ACTIVE_LINEAR);
        keys    = FcLayer(headDim, headDim, false, ACTIVE_LINEAR);
        queries = FcLayer(headDim, headDim, false, ACTIVE_LINEAR);
        fc_out  = FcLayer(headDim*headNum, embedSize, false, ACTIVE_LINEAR);
    }
    Tensor forward(const Tensor &values, const Tensor &keys, const Tensor &query, const Tensor &mask)
    {
        int N = query.shape[0];
        int value_len =  values.shape[1];
        int key_len = keys.shape[1];
        int query_len = query.shape[1];
        // split embeding into heads pieces
        values = values.reshape(N, value_len, headNum, headDim);
        keys = keys.reshape(N, key_len, headNum, headDim);
        queries = query.reshape(N, query_len, headNum, headDim);

        values = this->values(values);
        keys = this->keys(keys);
        queries = this->queries(queries);
        // query shape: (N, query_len, heads, heads_dim)
        // keys shape: (N, key_len, heads, heads_dim)
        // energy shape: (N, heads, query_len, key_len)
        Tensor energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]);

        if (mask.empty() == false) {
            energy = energy.masked_fill(mask == 0, float("-1e20"));
        }

        // attention shape: (N, heads, query_len, key_len)
        // values shape: (N, value_len, heads, heads_dim)
        // out shape: (N, query_len, heads, heads_dim) => (N, query_len, heads*heads_dim)
        attention = torch.softmax(energy/(self.embed_size**(1/2)), dim=3);

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim);
        out = fc_out(out);
        return out;
    }
    Tensor operator()(const Tensor &values, const Tensor &keys, const Tensor &query, const Tensor &mask)
    {
        return forward(values, keys, query, mask);
    }
};


class TransformerBlock
{
public:
    SelfAttention attention;
    LayerNorm norm1;
    LayerNorm norm2;
    FcLayer fc1;
    FcLayer fc2;
    Dropout dropout;
public:
    TransformerBlock(){}
    TransformerBlock(int embed_size, int heads, float dropout, int forward_expansion)
    {
        attention = SelfAttention(embed_size, heads)
        norm1 = LayerNorm(embed_size)
        norm2 = LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    }
    Tensor forward(Tensor &value, Tensor &key, Tensor &query, Tensor &mask):
    {
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    }
};
class Encoder
{
public:
    int embed_size;
    Tensor word_embedding;
    Tensor position_embedding;
    std::vector<TransformerBlock> layers;
    Dropout dropout;
public:
    Encoder(){}
    Encoder(
        int src_vocab_size,
        int embed_size,
        int num_layers,
        int heads,
        int forward_expansion,
        int dropout,
        int max_length)
    {
        word_embedding = nn.Embedding(src_vocab_size, embed_size)
        position_embedding = nn.Embedding(max_length, embed_size)
        layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion)
                                    for _ in range(num_layers)])
        dropout = nn.Dropout(dropout)

    }

    Tensor forward(Tesnor &x, Tesnor &mask)
    {
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    }
    Tensor operator()(Tesnor &x, Tesnor &mask)
    {
        return forward(x, mask);
    }

};
class DecoderBlock
{
public:
    SelfAttention attention;
    LayerNorm norm;
    TransformerBlock transformer_block;
    Dropout dropout;
public:
    DecoderBlock(){}
    DecoderBlock(int embed_size, int heads, int forward_expansion, int dropout)
    {
        attention = SelfAttention(embed_size, heads)
        norm = nn.LayerNorm(embed_size)
        transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        dropout = nn.Dropout(dropout)
    }

    Tensor forward(Tensor &x, Tensor &value, Tensor &key, Tensor &src_mask, Tensor &trg_mask)
    {
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    }
};

class Decoder
{
public:
    Tensor word_embedding;
    Tensor position_embedding;
    std::vector<DecoderBlock> layers;
    FcLayer fc_out;
    Dropout dropout;
public:
    Decoder(){}
    Decoder(
        int trg_vocab_size,
        int embed_size,
        int num_layers,
        int heads,
        int forward_expansion,
        int dropout,
        int max_length)
   {
        word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        position_embedding = nn.Embedding(max_length, embed_size)
        layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                                    for _ in range(num_layers)])
        fc_out = nn.Linear(embed_size, trg_vocab_size)
        dropout = nn.Dropout(dropout)

   }

   Tensor forward(Tensor &x, Tensor &enc_out, Tensor &src_mask, Tensor &trg_mask)
   {
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out
   }

   Tensor operator()(Tensor &x, Tensor &enc_out, Tensor &src_mask, Tensor &trg_mask)
   {
       return forward(x, enc_out, src_mask, trg_mask);
   }
};

class Transformer
{
public:
    Encoder encoder;
    Decoder decoder;
    int src_pad_idx;
    int trg_pad_idx;
public:
    Transformer(
        int src_vocab_size,
        int trg_vocab_size,
        int src_pad_idx,
        int trg_pad_idx,
        int embed_size=256,
        int num_layers=6,
        int forward_expansion=4,
        int heads=8,
        int dropout=0,
        int max_length=100)
    {
        encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        );
        decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        );

        this->src_pad_idx = src_pad_idx;
        this->trg_pad_idx = trg_pad_idx;

    }

    Tensor make_src_mask(Tensor &src)
    {
        Tensor src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        // (N, 1, 1, src_len)
        return src_mask;
    }

    Tensor make_trg_mask(Tensor &trg)
    {
        N, trg_len = trg.shape
        Tensor trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask;
    }

    Tensor forward(Tensor &src, Tensor &trg)
    {
        Tensor src_mask = make_src_mask(src);
        Tensor trg_mask = make_trg_mask(trg);
        Tensor enc_src = encoder(src, src_mask);
        return decoder(trg, enc_src, src_mask, trg_mask);
    }
};

#endif

#endif // TRANSFORMER_HPP
