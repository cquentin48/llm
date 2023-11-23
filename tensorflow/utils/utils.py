import keras_nlp

def init_lm():
    tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=256,
        add_end_token=True
    )
    return keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en", preprocessor=preprocessor)