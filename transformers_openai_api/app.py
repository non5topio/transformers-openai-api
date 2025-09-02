import json
import time
import torch
import logging
from typing import Any, Callable, Mapping, Optional
from flask import Flask, make_response, request, abort
from flask.json import jsonify
from functools import wraps
from .models import CausalLM, Model, Seq2Seq
from .metrics import Metrics

# Set up logging - Reduced verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)
models = {}
id = 0
metrics: Optional[Metrics]


def get_best_device() -> str:
    """Automatically detect the best available device: MPS (Mac) -> CPU"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    else:
        return 'cpu'


def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text using the model's tokenizer"""
    try:
        if model_name in models:
            model: Model = models[model_name]
            if hasattr(model, 'tokenizer'):
                tokens = model.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
        # Fallback: rough estimate if tokenizer not available
        return len(text.split()) * 1.3  # Approximate token count
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}, using word-based estimate")
        return int(len(text.split()) * 1.3)


def log_completion_details(model_name: str, prompt: str, response_content: str, usage: dict, request_id: str):
    """Log minimal completion information"""
    # Only log essential information at debug level
    pass  # Disabled verbose logging


def validate_request_data(data: dict, required_fields: list) -> tuple[bool, str]:
    """Validate request data and return (is_valid, error_message)"""
    if not data:
        return False, "Request body is empty"
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    return True, ""


def sanitize_chat_request(data: dict) -> dict:
    """Sanitize chat completion request parameters"""
    sanitized = data.copy()
    
    # Validate and sanitize max_tokens - INCREASED TO SUPPORT 124K
    max_tokens = sanitized.get('max_tokens', 1000)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        sanitized['max_tokens'] = 1000
        logger.warning(f"Invalid max_tokens {max_tokens}, using 1000")
    elif max_tokens > 124000:  # Allow up to 124k tokens
        sanitized['max_tokens'] = 124000
        logger.warning(f"max_tokens {max_tokens} too high, clamping to 124000")
    
    # Sanitize temperature
    temperature = sanitized.get('temperature', 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0:
        sanitized['temperature'] = 0.7
        logger.warning(f"Invalid temperature {temperature}, using 0.7")
    elif temperature > 2.0:
        sanitized['temperature'] = 2.0
        logger.warning(f"Temperature {temperature} too high, clamping to 2.0")
    
    # Sanitize top_p
    top_p = sanitized.get('top_p', 1.0)
    if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1.0:
        sanitized['top_p'] = 0.9
        logger.warning(f"Invalid top_p {top_p}, using 0.9")
    
    # Validate messages
    messages = sanitized.get('messages', [])
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Messages must be a non-empty list")
    
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {i} must be a dictionary")
        if 'role' not in message or 'content' not in message:
            raise ValueError(f"Message {i} must have 'role' and 'content' fields")
        if message['role'] not in ['system', 'user', 'assistant']:
            raise ValueError(f"Message {i} has invalid role: {message['role']}")
    
    return sanitized


def check_token(f: Callable):
    @wraps(f)
    def decorator(*args, **kwargs):
        bearer_tokens = app.config.get('BEARER_TOKENS')
        if bearer_tokens is None:
            return f(*args, **kwargs)

        authorization = request.headers.get('Authorization')
        if not authorization:
            return make_response(jsonify({
                'error': {
                    'message': 'Authorization header required',
                    'type': 'invalid_request_error',
                    'code': 'missing_authorization'
                }
            }), 401)
            
        if authorization.startswith('Bearer '):
            token = authorization[7:]
            if token in bearer_tokens:
                return f(*args, **kwargs)
        return make_response(jsonify({
            'error': {
                'message': 'Invalid token',
                'type': 'invalid_request_error',
                'code': 'invalid_token'
            }
        }), 401)
    return decorator


def convert_model_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    config = {}
    if val is not None:
        for key, value in val.items():
            if key == 'torch_dtype':
                if value == 'float16':
                    config['torch_dtype'] = torch.float16
                elif value == 'float32':
                    config['torch_dtype'] = torch.float32
                elif value == 'int8':
                    config['torch_dtype'] = torch.int8
                else:
                    raise RuntimeError(
                        f"Unknown torch_dtype {config['torch_dtype']}")
            else:
                config[key] = value
    return config


def convert_tokenizer_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return val if val is not None else {}


def convert_generate_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    config = {}
    if val is not None:
        for key, value in val.items():
            if key == 'max_tokens':
                config['max_new_tokens'] = value  # Use max_new_tokens instead of max_length
            elif key == 'stream':
                continue
            elif key in ['n', 'stop']:
                continue
            else:
                config[key] = value
    return config


def convert_decode_config(val: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return val if val is not None else {}


def completion(model_name: str):
    global id
    this_id = id
    id += 1

    try:
        if not request.json:
            return make_response(jsonify({
                'error': {
                    'message': 'Request body must be JSON',
                    'type': 'invalid_request_error',
                    'code': 'invalid_json'
                }
            }), 400)

        # Extract prompt for logging
        prompt = request.json.get('prompt', '')
        if isinstance(prompt, list):
            prompt = ' '.join(prompt)

        model: Model = models[model_name]
        response = model.completions(convert_generate_config(request.json))
        response.update({
            'object': 'text_completion',
            'model': model_name,
            'created': int(time.time()),
            'id': f'cmpl-{this_id}'
        })

        # Extract generated text for logging
        generated_text = ""
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            generated_text = choice.get('text', str(choice))

        # Log completion details
        log_completion_details(model_name, prompt, generated_text, response.get('usage', {}), f'cmpl-{this_id}')

        global metrics
        if metrics is not None:
            metrics.update(response)

        return make_response(jsonify(response))
        
    except Exception as e:
        logger.error(f"Error in completion: {e}")
        return make_response(jsonify({
            'error': {
                'message': str(e),
                'type': 'internal_server_error',
                'code': 'server_error'
            }
        }), 500)


@app.route('/v1/engines')
def v1_engines():
    return make_response(jsonify({
        'data': [{
            'object': 'engine',
            'id': model_id,
            'ready': True,
            'owner': 'openai',
            'permissions': None,
            'created': None
        } for model_id in models.keys()]
    }))


@app.route('/v1/completions', methods=['POST'])
@check_token
def v1_completions():
    try:
        if not request.json or 'model' not in request.json:
            return make_response(jsonify({
                'error': {
                    'message': 'Model parameter is required',
                    'type': 'invalid_request_error',
                    'code': 'missing_model'
                }
            }), 400)
        return completion(request.json['model'])
    except Exception as e:
        logger.error(f"Error in v1_completions: {e}")
        return make_response(jsonify({
            'error': {
                'message': str(e),
                'type': 'internal_server_error',
                'code': 'server_error'
            }
        }), 500)


@app.route('/v1/engines/<model_name>/completions', methods=['POST'])
@check_token
def engine_completion(model_name: str):
    return completion(model_name)


@app.route('/v1/chat/completions', methods=['POST'])
@check_token
def v1_chat_completions():
    try:
        # Validate request
        if not request.json:
            return make_response(jsonify({
                'error': {
                    'message': 'Request body must be JSON',
                    'type': 'invalid_request_error',
                    'code': 'invalid_json'
                }
            }), 400)

        # Validate required fields
        is_valid, error_msg = validate_request_data(request.json, ['model', 'messages'])
        if not is_valid:
            return make_response(jsonify({
                'error': {
                    'message': error_msg,
                    'type': 'invalid_request_error',
                    'code': 'invalid_request'
                }
            }), 400)

        # Sanitize request data
        try:
            data = sanitize_chat_request(request.json)
        except ValueError as ve:
            return make_response(jsonify({
                'error': {
                    'message': str(ve),
                    'type': 'invalid_request_error',
                    'code': 'invalid_request'
                }
            }), 400)

        model_name = data.get('model')
        
        if not model_name or model_name not in models:
            return make_response(jsonify({
                'error': {
                    'message': f'Model {model_name} not found. Available models: {list(models.keys())}',
                    'type': 'invalid_request_error',
                    'code': 'model_not_found'
                }
            }), 404)
        
        messages = data.get('messages', [])
        
        # Convert messages to a single prompt with better formatting
        prompt_parts = []
        for message in messages:
            role = message['role']
            content = message['content'].strip()
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        # Create completion request with validated parameters
        completion_request = {
            'prompt': prompt,
            'max_tokens': data.get('max_tokens', 100),
            'temperature': data.get('temperature', 0.7),
            'top_p': data.get('top_p', 0.9),
            'n': data.get('n', 1),
            'stream': data.get('stream', False),
            'stop': data.get('stop', None)
        }
        
        # Generate response
        global id
        this_id = id
        id += 1

        model: Model = models[model_name]
        response = model.completions(convert_generate_config(completion_request))
        
        # Process response
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            
            # Extract content from various possible formats
            if 'text' in choice:
                content = choice['text'].strip()
            elif 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content'].strip()
            else:
                content = str(choice).strip()
            
            # Remove the original prompt if it's being echoed back
            # Find where the actual response starts after the prompt
            original_prompt = prompt.strip()
            if content.startswith(original_prompt):
                # Remove the echoed prompt and get only the new generated content
                content = content[len(original_prompt):].strip()
            
            # Remove "Assistant:" prefix if it exists at the start of the response
            if content.startswith('Assistant:'):
                content = content[10:].strip()
            
            # If there's still conversation history being echoed, try to extract just the final response
            if 'Human:' in content or 'System:' in content:
                # Split by Assistant: and take the last part
                parts = content.split('Assistant:')
                if len(parts) > 1:
                    content = parts[-1].strip()
                else:
                    # Try to find the last meaningful response by looking for YAML or other patterns
                    lines = content.split('\n')
                    # Look for lines that start with yaml block or YAML content
                    yaml_start = -1
                    for i, line in enumerate(lines):
                        if line.strip().startswith('```yaml') or line.strip().startswith('language:'):
                            yaml_start = i
                            break
                    
                    if yaml_start >= 0:
                        content = '\n'.join(lines[yaml_start:]).strip()
            
            finish_reason = choice.get('finish_reason', 'stop')
        else:
            content = "I apologize, but I was unable to generate a response."
            finish_reason = 'error'
        
        # Create chat completion response
        chat_response = {
            'id': f'chatcmpl-{this_id}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model_name,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content
                },
                'finish_reason': finish_reason
            }],
            'usage': response.get('usage', {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            })
        }

        global metrics
        if metrics is not None:
            metrics.update(chat_response)

        return make_response(jsonify(chat_response))
        
    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        return make_response(jsonify({
            'error': {
                'message': f'Internal server error: {str(e)}',
                'type': 'internal_server_error',
                'code': 'server_error'
            }
        }), 500)


@app.route('/v1/metrics')
def metrics_():
    global metrics
    if metrics is None:
        abort(404)

    return make_response(jsonify(metrics.get()))


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({
        'error': {
            'message': 'Endpoint not found',
            'type': 'invalid_request_error',
            'code': 'not_found'
        }
    }), 404)


@app.errorhandler(500)
def internal_error(error):
    return make_response(jsonify({
        'error': {
            'message': 'Internal server error',
            'type': 'internal_server_error',
            'code': 'server_error'
        }
    }), 500)


def make_transformers_openai_api(config_path: str) -> Flask:
    try:
        app.config.from_file(config_path, load=json.load)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

    if app.config.get('METRICS', 1) != 0:
        global metrics
        metrics = Metrics()

    # Get the best available device for auto-detection
    best_device = get_best_device()
    logger.info(f"Auto-detected device: {best_device}")

    for mapping, config in app.config['MODELS'].items():
        if config.get('ENABLED', True) == False:
            continue
            
        try:
            model_config = convert_model_config(config.get('MODEL_CONFIG'))
            
            # Use auto-detected device if not explicitly set or if set to 'auto'
            model_device = config.get('MODEL_DEVICE')
            if model_device is None or model_device == 'auto':
                # Only auto-detect if device_map is not set to 'auto' (for accelerate)
                if not (isinstance(model_config, dict) and model_config.get('device_map') == 'auto'):
                    model_device = best_device
            
            tokenizer_config = convert_tokenizer_config(
                config.get('TOKENIZER_CONFIG'))
            
            # Use auto-detected device for tokenizer if not explicitly set or if set to 'auto'
            tokenizer_device = config.get('TOKENIZER_DEVICE')
            if tokenizer_device is None or tokenizer_device == 'auto':
                tokenizer_device = best_device
            
            generate_config = convert_generate_config(
                config.get('GENERATE_CONFIG'))
            decode_config = convert_decode_config(
                config.get('DECODE_CONFIG'))
            
            logger.info(f"Loading model '{mapping}' on device: model={model_device}, tokenizer={tokenizer_device}")
            
            if config['TYPE'] == 'Seq2Seq':
                models[mapping] = Seq2Seq(
                    config['NAME'], model_config, model_device, tokenizer_config, tokenizer_device, generate_config, decode_config)
            elif config['TYPE'] == 'CausalLM':
                models[mapping] = CausalLM(
                    config['NAME'], model_config, model_device, tokenizer_config, tokenizer_device, generate_config, decode_config)
            else:
                raise RuntimeError(f'Unknown model type {config["TYPE"]}')
                
            logger.info(f"Successfully loaded model '{mapping}'")
            
        except Exception as e:
            logger.error(f"Failed to load model '{mapping}': {e}")
            raise

    return app
