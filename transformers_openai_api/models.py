from abc import ABC
import logging
import time
import torch
from typing import Any, List, Mapping, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_generation_params(params: dict) -> dict:
    """Sanitize generation parameters to prevent numerical instability."""
    sanitized = params.copy()
    
    # Clamp temperature to reasonable bounds
    if 'temperature' in sanitized:
        temp = float(sanitized['temperature'])
        if temp <= 0 or temp != temp:  # Handle 0, negative, or NaN
            sanitized['temperature'] = 0.7
            logger.warning(f"Invalid temperature {temp}, using 0.7")
        elif temp > 2.0:
            sanitized['temperature'] = 2.0
            logger.warning(f"Temperature {temp} too high, clamping to 2.0")
        else:
            sanitized['temperature'] = max(0.01, min(temp, 2.0))
    
    # Clamp top_p to valid range
    if 'top_p' in sanitized:
        top_p = float(sanitized['top_p'])
        if top_p <= 0 or top_p > 1.0 or top_p != top_p:  # Handle invalid or NaN
            sanitized['top_p'] = 0.9
            logger.warning(f"Invalid top_p {top_p}, using 0.9")
        else:
            sanitized['top_p'] = max(0.01, min(top_p, 1.0))
    
    # Validate top_k
    if 'top_k' in sanitized:
        top_k = sanitized['top_k']
        if not isinstance(top_k, int) or top_k < 0:
            sanitized['top_k'] = 50
            logger.warning(f"Invalid top_k {top_k}, using 50")
        elif top_k > 1000:
            sanitized['top_k'] = 1000
            logger.warning(f"top_k {top_k} too high, clamping to 1000")
    
    # Validate max_length/max_new_tokens - INCREASED LIMITS FOR 124K CONTEXT
    if 'max_length' in sanitized:
        max_len = sanitized['max_length']
        if not isinstance(max_len, int) or max_len <= 0:
            sanitized['max_length'] = 1000
            logger.warning(f"Invalid max_length {max_len}, using 1000")
        elif max_len > 124000:  # Allow up to 124k tokens
            sanitized['max_length'] = 124000
            logger.warning(f"max_length {max_len} too high, clamping to 124000")
    
    if 'max_new_tokens' in sanitized:
        max_new = sanitized['max_new_tokens']
        if not isinstance(max_new, int) or max_new <= 0:
            sanitized['max_new_tokens'] = 1000
            logger.warning(f"Invalid max_new_tokens {max_new}, using 1000")
        elif max_new > 124000:  # Allow up to 124k tokens
            sanitized['max_new_tokens'] = 124000
            logger.warning(f"max_new_tokens {max_new} too high, clamping to 124000")
    
    # Add attention mask and pad token handling
    sanitized['pad_token_id'] = sanitized.get('pad_token_id', 0)
    sanitized['use_cache'] = True
    
    return sanitized


def get_prompts(request: Mapping[str, Any]) -> List[str]:
    prompt = request['prompt']
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt


def _completions_auto(
        request: Mapping[str, Any],
        tokenizer: Any,
        tokenizer_device: Optional[str],
        model: Any,
        generate_config: Mapping[str, Any],
        decode_config: Mapping[str, Any],
        auto_echo: bool):
    
    try:
        generate_args = {}
        generate_args.update(generate_config)
        generate_args.update(request)
        
        # Sanitize parameters to prevent numerical issues
        generate_args = sanitize_generation_params(generate_args)

        decode_args = {
            "skip_special_tokens": True
        }
        decode_args.update(decode_config)

        # Set up sampling parameters more carefully
        if ('top_p' in generate_args or 'top_k' in generate_args or 'temperature' in generate_args) and 'do_sample' not in generate_args:
            generate_args['do_sample'] = True
            
            # Handle temperature edge cases
            temp = generate_args.get('temperature', 1.0)
            if temp <= 0.01:  # Very low temperature - use greedy decoding
                generate_args['do_sample'] = False
                generate_args.pop('temperature', None)
                generate_args.pop('top_p', None)
                generate_args.pop('top_k', None)
            else:
                # Ensure we have reasonable defaults
                if 'top_k' not in generate_args:
                    generate_args['top_k'] = 50
                if 'top_p' not in generate_args:
                    generate_args['top_p'] = 0.9

        prompts = get_prompts(generate_args)
        echo = generate_args.get('echo', False)
        n = generate_args.get('n', 1)

        # Remove API-specific parameters that aren't used by transformers
        for param in ['model', 'prompt', 'n', 'best_of', 'presence_penalty', 
                     'frequency_penalty', 'logit_bias', 'stop', 'stream']:
            generate_args.pop(param, None)

        inputs = []
        prompt_tokens_count = 0
        
        # Process prompts with better error handling - INCREASED LIMITS
        for i, prompt in enumerate(prompts):
            try:
                # Log input details
                logger.debug(f"Processing prompt {i+1}/{len(prompts)}: {len(prompt)} characters")
                
                # Increased prompt length limit for 124k context
                if len(prompt) > 500000:  # ~124k tokens worth of characters
                    prompt = prompt[:500000]
                    logger.warning("Prompt truncated to 500000 characters")
                
                # Increased tokenization limits
                input_tokens = tokenizer(prompt, return_tensors="pt", 
                                       truncation=True, max_length=120000)  # Near 124k limit
                input_ids = input_tokens.input_ids
                
                # Create attention mask if not present
                if 'attention_mask' not in input_tokens:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = input_tokens.attention_mask
                
                if tokenizer_device is not None:
                    input_ids = input_ids.to(tokenizer_device)
                    attention_mask = attention_mask.to(tokenizer_device)
                
                prompt_token_count = input_ids.size(dim=1)
                prompt_tokens_count += prompt_token_count
                
                # Log tokenization details
                logger.debug(f"Prompt {i+1} tokenized: {prompt_token_count} tokens")
                
                inputs.append({'input_ids': input_ids, 'attention_mask': attention_mask})
                
            except Exception as e:
                logger.error(f"Error processing prompt {i+1}: {e}")
                raise ValueError(f"Failed to process prompt {i+1}: {str(e)}")

        choices = []
        completion_tokens_count = 0
        
        logger.info(f"Starting generation for {len(inputs)} prompts with {n} completions each")
        
        # Generate responses with robust error handling
        for i in range(len(inputs)):
            for attempt in range(n):
                try:
                    logger.debug(f"Generating completion {attempt+1}/{n} for prompt {i+1}")
                    
                    # Set attention mask in generation args
                    gen_kwargs = generate_args.copy()
                    gen_kwargs['attention_mask'] = inputs[i]['attention_mask']
                    
                    # Add additional stability parameters
                    gen_kwargs['do_sample'] = gen_kwargs.get('do_sample', True)
                    gen_kwargs['early_stopping'] = True
                    gen_kwargs['repetition_penalty'] = gen_kwargs.get('repetition_penalty', 1.1)
                    
                    # Log generation parameters
                    max_new_tokens = gen_kwargs.get('max_new_tokens', gen_kwargs.get('max_length', 1000))
                    logger.debug(f"Generation params: max_new_tokens={max_new_tokens}, "
                               f"temperature={gen_kwargs.get('temperature', 'N/A')}, "
                               f"do_sample={gen_kwargs.get('do_sample', False)}")
                    
                    # Generate with error catching
                    generation_start = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs[i]['input_ids'], 
                            **gen_kwargs
                        )
                    generation_time = time.time() - generation_start
                    
                    # Handle output processing
                    if len(outputs.shape) > 1:
                        output = outputs[0]
                    else:
                        output = outputs
                    
                    # Calculate actual completion tokens
                    actual_completion_tokens = len(output) - inputs[i]['input_ids'].size(1)
                    completion_tokens_count += actual_completion_tokens
                    
                    # Log generation results
                    logger.debug(f"Generated {actual_completion_tokens} tokens in {generation_time:.2f}s "
                               f"({actual_completion_tokens/generation_time:.1f} tokens/s)")
                    
                    # Decode the output
                    text = tokenizer.decode(output, **decode_args)
                    
                    # Remove the original prompt from the output for CausalLM
                    if not auto_echo and not echo:
                        original_prompt = tokenizer.decode(inputs[i]['input_ids'][0], **decode_args)
                        if text.startswith(original_prompt):
                            text = text[len(original_prompt):].strip()
                    
                    if echo and not auto_echo:
                        text = prompts[i] + text
                    
                    # Log the actual generated text preview
                    text_preview = text[:100] + "..." if len(text) > 100 else text
                    logger.debug(f"Generated text preview: {text_preview}")
                    
                    choices.append({
                        'text': text,
                        'index': i,
                        'finish_reason': 'stop'
                    })
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error("CUDA out of memory during generation")
                    torch.cuda.empty_cache()
                    choices.append({
                        'text': "Error: Out of memory",
                        'index': i,
                        'finish_reason': 'error'
                    })
                except Exception as e:
                    logger.error(f"Error during generation attempt {attempt+1}: {e}")
                    if "probability tensor contains" in str(e) or "inf" in str(e) or "nan" in str(e):
                        # Retry with more conservative parameters but still allow high token counts
                        try:
                            logger.info("Retrying with conservative parameters due to numerical instability")
                            conservative_args = {
                                'max_new_tokens': min(gen_kwargs.get('max_new_tokens', 1000), 10000),  # Conservative but still generous
                                'do_sample': False,  # Use greedy decoding
                                'attention_mask': inputs[i]['attention_mask'],
                                'pad_token_id': tokenizer.eos_token_id or 0
                            }
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    inputs[i]['input_ids'],
                                    **conservative_args
                                )
                            
                            output = outputs[0] if len(outputs.shape) > 1 else outputs
                            actual_completion_tokens = len(output) - inputs[i]['input_ids'].size(1)
                            completion_tokens_count += actual_completion_tokens
                            
                            text = tokenizer.decode(output, **decode_args)
                            
                            # Remove prompt from output
                            if not auto_echo and not echo:
                                original_prompt = tokenizer.decode(inputs[i]['input_ids'][0], **decode_args)
                                if text.startswith(original_prompt):
                                    text = text[len(original_prompt):].strip()
                            
                            choices.append({
                                'text': text,
                                'index': i,
                                'finish_reason': 'stop'
                            })
                            logger.info(f"Successfully recovered with conservative parameters, generated {actual_completion_tokens} tokens")
                            
                        except Exception as retry_e:
                            logger.error(f"Retry also failed: {retry_e}")
                            choices.append({
                                'text': f"Error: Unable to generate response due to numerical instability",
                                'index': i,
                                'finish_reason': 'error'
                            })
                    else:
                        choices.append({
                            'text': f"Error: {str(e)}",
                            'index': i,
                            'finish_reason': 'error'
                        })

        # Log final token usage summary
        logger.info(f"Token usage summary - Input: {prompt_tokens_count}, Output: {completion_tokens_count}, Total: {prompt_tokens_count + completion_tokens_count}")

        return {
            'choices': choices,
            'usage': {
                'prompt_tokens': prompt_tokens_count,
                'completion_tokens': completion_tokens_count,
                'total_tokens': prompt_tokens_count + completion_tokens_count
            }
        }
        
    except Exception as e:
        logger.error(f"Critical error in _completions_auto: {e}")
        return {
            'choices': [{
                'text': f"Critical error: {str(e)}",
                'index': 0,
                'finish_reason': 'error'
            }],
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        }


class Model(ABC):

    def completions(self, request: Mapping[str, Any]):
        pass


class Seq2Seq(Model):
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    generate_config: Mapping[str, Any]
    decode_config: Mapping[str, Any]
    tokenizer_device: Optional[str]

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            model_config: Mapping[str, Any],
            model_device: Optional[str],
            tokenizer_config: Mapping[str, Any],
            tokenizer_device: Optional[str],
            generate_config: Mapping[str, Any],
            decode_config: Mapping[str, Any]) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path, **model_config)
        if model_device is not None:
            self.model = self.model.to(model_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_config)
        self.generate_config = generate_config
        self.decode_config = decode_config
        self.tokenizer_device = tokenizer_device

    def completions(self, request) -> List[str]:
        return _completions_auto(request, self.tokenizer, self.tokenizer_device, self.model, self.generate_config, self.decode_config, False)


class CausalLM(Model):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generate_config: Mapping[str, Any]
    decode_config: Mapping[str, Any]
    tokenizer_device: Optional[str]

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            model_config: Mapping[str, Any],
            model_device: Optional[str],
            tokenizer_config: Mapping[str, Any],
            tokenizer_device: Optional[str],
            generate_config: Mapping[str, Any],
            decode_config: Mapping[str, Any]) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **model_config)
        if model_device is not None:
            self.model = self.model.to(model_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_config)
        self.generate_config = generate_config
        self.decode_config = decode_config
        self.tokenizer_device = tokenizer_device

    def completions(self, request) -> List[str]:
        return _completions_auto(request, self.tokenizer, self.tokenizer_device, self.model, self.generate_config, self.decode_config, True)
