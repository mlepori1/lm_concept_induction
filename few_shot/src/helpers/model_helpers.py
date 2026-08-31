import torch

###
#   This file implements basic LM operations, and also provides handles
#   for steering.
###

def get_token_id(word, tokenizer):
    """Return the single token ID for a word (with a leading space)."""
    return tokenizer.encode(" " + word.strip(), add_special_tokens=False)[0]


def get_hidden_states_and_logprobs_last_token(input_str, tokenizer, model, context_id):
    """Run the model on input_str and return both all-layer hidden states and the
    next-token log-probability distribution, both read at the last position.

    The 'last position' is where the model accumulates context from everything that
    came before, so it's the natural place to read out the concept representation.
    Callers that only need the hidden states can ignore the log_probs return value.

    Returns:
        hidden_states: (n_layers, d) tensor.
        log_probs: (vocab_size,) tensor, log-softmax of the logits predicting whatever
            token would follow input_str.
    """
    if not input_str.startswith(" "):
        input_str = " " + input_str
    token_ids = tokenizer.encode(input_str, add_special_tokens=False)
    input_ids = torch.tensor([[context_id] + token_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = torch.stack(
        [layer_hs[0, -1, :].cpu().float() for layer_hs in outputs.hidden_states]
    )
    log_probs = torch.log_softmax(outputs.logits[0, -1, :], dim=-1).cpu().float()
    return hidden_states, log_probs


def make_forward_hook(steering_vector, magnitude, device="cuda"):
    """Return a PyTorch forward hook for steering"""
    injection = (magnitude * steering_vector).to(device)

    def hook(module, layer_input, layer_output):
        # layer_output is either a tensor or a tuple whose first element is the tensor
        if isinstance(layer_output, tuple):
            layer_output[0][
                :, -1, :
            ] += injection  # last token position, all batch items
        else:
            layer_output[:, -1, :] += injection
        return layer_output

    return hook


def get_next_token_log_probs(
    input_ids, model, model_layers, steering_vector=None, magnitude=0.0, steer_layer=None, device="cuda"
):
    """Run the model and return the log-probability distribution over next tokens.

    Optionally injects a steering vector at steer_layer via a forward hook.
    The hook is registered before the forward pass and removed immediately after.

    Returns a (vocab_size,) tensor of log-probabilities.
    """
    input_tensor = torch.tensor([input_ids], device=device)

    hook_handle = None
    if steering_vector is not None:
        hook_handle = model_layers[steer_layer].register_forward_hook(
            make_forward_hook(steering_vector, magnitude, device)
        )

    with torch.no_grad():
        output = model(input_tensor)

    if hook_handle is not None:
        hook_handle.remove()
    return torch.log_softmax(output.logits[0, -1, :], dim=-1)
