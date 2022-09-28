import fairseq
from fairseq.models import register_model_architecture

from .models import encoder_decoder_with_domain_disc, encoder_decoder_with_domain_routing, \
    encoder_decoder_with_domain_disc_gru, encoder_decoder_with_domain_routing_gru, \
    encoder_decoder_with_adapter, encoder_decoder_with_adapter_test, adapter_transformer, \
    encoder_decoder_with_language_domain_adapter
from .tasks import translation_multilingual_multidomain
from .criterions import domain_aware_cross_entropy_with_label_smoothing, label_smoothed_cross_entropy_posterior_regularization, \
    label_smoothed_cross_entropy_fusion_regularization

@register_model_architecture("transformer", "toy_transformer")
def toy_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 128)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    fairseq.models.transformer.base_architecture(args)

@register_model_architecture("domain_aware_transformer", "domain_aware_transformer")
def domain_aware_transformer(args):
    fairseq.models.transformer.base_architecture(args)

@register_model_architecture("domain_aware_transformer", "domain_aware_mbart_large")
def domain_aware_mbart_large(args):
    fairseq.models.bart.bart_large_architecture(args)
    
@register_model_architecture("domain_aware_transformer", "domain_agnostic_mbart_large")
def domain_agnostic_mbart_large(args):
    args.gradient_reverse = True
    fairseq.models.bart.bart_large_architecture(args)

@register_model_architecture("domain_adapter_transformer", "domain_adapter_transformer")
def domain_adapter_transformer(args):
    args.down_sample = getattr(args, "down_sample", None)
    args.encoder_adapter = getattr(args, "encoder_adapter", False)
    args.decoder_adapter = getattr(args, "decoder_adapter", False)
    fairseq.models.bart.bart_large_architecture(args)

@register_model_architecture("domain_adapter_transformer", "domain_adapter_mbart_large")
def domain_adapter_mbart_large(args):
    args.down_sample = getattr(args, "down_sample", None)
    args.encoder_adapter = getattr(args, "encoder_adapter", False)
    args.decoder_adapter = getattr(args, "decoder_adapter", True)
    args.freeze_encoder = getattr(args, "freeze_encoder", True)
    args.freeze_decoder = getattr(args, "freeze_decoder", True)
    fairseq.models.bart.bart_large_architecture(args)


@register_model_architecture("domain_aware_transformer", "toy_domain_aware_transformer")
def toy_domain_aware_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 128)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    fairseq.models.transformer.base_architecture(args)

@register_model_architecture("domain_adapter_transformer", "toy_domain_adapter_transformer")
def toy_domain_aware_transformer(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 128)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.down_sample = getattr(args, "down_sample", 64)
    # args.encoder_adapter = getattr(args, "encoder_adapter", False)
    # args.decoder_adapter = getattr(args, "decoder_adapter", True)
    fairseq.models.transformer.base_architecture(args)
